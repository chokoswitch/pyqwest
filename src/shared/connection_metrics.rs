use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use hyper_util::client::legacy::{ConnectionInfo, PoolMetrics};
use pyo3::{
    pyclass, pymethods,
    sync::{MutexExt as _, PyOnceLock},
    types::{PyAnyMethods as _, PyDict, PyDictMethods as _, PyList, PyListMethods as _, PyString},
    Bound, Py, PyAny, PyResult, Python,
};

use crate::shared::{constants::Constants, otel::network_protocol_version};

// Not clear if metrics can be GC'd and disappear, we keep a reference in case.
struct ConnectionMetrics {
    /// The pool metrics of every transport, reported by one callback. The
    /// OpenTelemetry Python SDK keeps only the first callback registered for
    /// an instrument name, so the transports cannot each register their own.
    pools: Arc<Mutex<Vec<PoolMetrics>>>,
    _open_connections: Py<PyAny>,
}

static CONNECTION_METRICS: PyOnceLock<ConnectionMetrics> = PyOnceLock::new();

/// Returns a new handle for a transport's connection pool to report through
/// and includes it in the `http.client.open_connections` metric. Like the
/// runtime metrics, the metric is registered once per process, with the first
/// meter.
pub(crate) fn start_connection_metrics(
    py: Python<'_>,
    meter: &Bound<'_, PyAny>,
    constants: &Constants,
) -> PyResult<PoolMetrics> {
    let metrics = CONNECTION_METRICS.get_or_try_init(py, || {
        let pools = Arc::new(Mutex::new(Vec::new()));
        // Inline strings are fine here since we are inside a PyOnceLock.
        let open_connections = meter.call_method1(
            &constants.create_observable_up_down_counter,
            (
                "http.client.open_connections",
                (OpenConnectionsCallback {
                    pools: pools.clone(),
                    http_connection_state: PyString::new(py, "http.connection.state").unbind(),
                    active: PyString::new(py, "active").unbind(),
                    idle: PyString::new(py, "idle").unbind(),
                    network_peer_address: PyString::new(py, "network.peer.address").unbind(),
                    constants: constants.clone(),
                },),
                "{connection}",
                "Number of outbound HTTP connections that are currently active or idle on the client.",
            ),
        )?;
        Ok::<_, pyo3::PyErr>(ConnectionMetrics {
            pools,
            _open_connections: open_connections.unbind(),
        })
    })?;
    let pool_metrics = PoolMetrics::new();
    metrics
        .pools
        .lock_py_attached(py)
        .unwrap()
        .push(pool_metrics.clone());
    Ok(pool_metrics)
}

#[pyclass(module = "_pyqwest.otel", name = "OpenConnectionsCallback", frozen)]
struct OpenConnectionsCallback {
    pools: Arc<Mutex<Vec<PoolMetrics>>>,
    http_connection_state: Py<PyString>,
    active: Py<PyString>,
    idle: Py<PyString>,
    network_peer_address: Py<PyString>,
    constants: Constants,
}

impl OpenConnectionsCallback {
    fn attributes<'py>(
        &self,
        py: Python<'py>,
        connection: &ConnectionInfo,
    ) -> PyResult<Bound<'py, PyDict>> {
        let constants = &self.constants;
        let default_port = if connection.scheme == http::uri::Scheme::HTTPS {
            443
        } else {
            80
        };
        let attrs = PyDict::new(py);
        attrs.set_item(&constants.server_address, connection.authority.host())?;
        attrs.set_item(
            &constants.server_port,
            connection.authority.port_u16().unwrap_or(default_port),
        )?;
        attrs.set_item(
            &constants.network_protocol_version,
            network_protocol_version(py, connection.version, constants),
        )?;
        if let Some(peer) = connection.peer {
            attrs.set_item(&self.network_peer_address, peer.ip().to_string())?;
        }
        Ok(attrs)
    }
}

#[pymethods]
impl OpenConnectionsCallback {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        _options: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        // (active, idle) summed over the transports, since several may connect
        // to the same server. Handles whose client is gone are dropped.
        let mut counts: HashMap<ConnectionInfo, (usize, usize)> = HashMap::new();
        self.pools.lock_py_attached(py).unwrap().retain(|pool| {
            let Some(open_connections) = pool.open_connections() else {
                return false;
            };
            for open in open_connections {
                let count = counts.entry(open.connection).or_default();
                count.0 += open.active;
                count.1 += open.idle;
            }
            true
        });

        let observation_class = self.constants.observation_class.bind(py);
        let observations = PyList::empty(py);
        for (connection, (active, idle)) in &counts {
            for (state, value) in [(&self.active, *active), (&self.idle, *idle)] {
                let attrs = self.attributes(py, connection)?;
                attrs.set_item(&self.http_connection_state, state)?;
                observations.append(observation_class.call1((value, attrs))?)?;
            }
        }
        Ok(observations)
    }
}
