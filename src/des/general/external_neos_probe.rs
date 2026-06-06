//! Read-only readiness probe for the public NEOS XML-RPC service.
//!
//! Submitting optimization jobs to NEOS requires user configuration such as a
//! valid email address. This module deliberately avoids submission and only
//! checks the documented `ping()` method over the current HTTPS XML-RPC endpoint.

use native_tls::TlsConnector;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const DEFAULT_NEOS_XMLRPC_URL: &str = "https://neos-server.org:3333/";
const NEOS_PING_XML: &str = r#"<?xml version="1.0"?><methodCall><methodName>ping</methodName><params></params></methodCall>"#;
const NEOS_ENDPOINT_ENV_NAMES: &[&str] = &["ORES_NEOS_XMLRPC_URL", "NEOS_XMLRPC_URL"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalNeosProbe {
    pub ready: bool,
    pub endpoint: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NeosXmlRpcEndpoint {
    url: String,
    host: String,
    port: u16,
    path: String,
}

pub fn probe_external_neos_server(timeout_ms: u64) -> ExternalNeosProbe {
    let endpoint = match configured_neos_xmlrpc_endpoint() {
        Ok(endpoint) => endpoint,
        Err(err) => {
            return ExternalNeosProbe {
                ready: false,
                endpoint: DEFAULT_NEOS_XMLRPC_URL.to_string(),
                message: err,
            };
        }
    };
    match run_neos_ping(&endpoint, timeout_ms) {
        Ok(response) => classify_neos_ping_response(&endpoint, &response),
        Err(err) => ExternalNeosProbe {
            ready: false,
            endpoint: endpoint.url,
            message: err,
        },
    }
}

fn configured_neos_xmlrpc_endpoint() -> Result<NeosXmlRpcEndpoint, String> {
    let value = NEOS_ENDPOINT_ENV_NAMES
        .iter()
        .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_NEOS_XMLRPC_URL.to_string());
    parse_neos_xmlrpc_endpoint(&value)
}

fn parse_neos_xmlrpc_endpoint(value: &str) -> Result<NeosXmlRpcEndpoint, String> {
    let trimmed = value.trim();
    let rest = trimmed
        .strip_prefix("https://")
        .ok_or_else(|| "NEOS XML-RPC endpoint must start with https://".to_string())?;
    let (authority, path) = rest
        .split_once('/')
        .map(|(authority, path)| (authority, format!("/{path}")))
        .unwrap_or((rest, "/".to_string()));
    if authority.is_empty() {
        return Err("NEOS XML-RPC endpoint is missing a host".to_string());
    }
    if authority.contains('@') {
        return Err("NEOS XML-RPC endpoint must not include credentials".to_string());
    }
    let (host, port) = authority
        .rsplit_once(':')
        .map(|(host, port)| {
            let port = port
                .parse::<u16>()
                .map_err(|_| format!("invalid NEOS XML-RPC port '{port}'"))?;
            Ok::<_, String>((host.to_string(), port))
        })
        .unwrap_or_else(|| Ok((authority.to_string(), 443)))?;
    if host.is_empty() {
        return Err("NEOS XML-RPC endpoint is missing a host".to_string());
    }
    Ok(NeosXmlRpcEndpoint {
        url: trimmed.to_string(),
        host,
        port,
        path,
    })
}

fn run_neos_ping(endpoint: &NeosXmlRpcEndpoint, timeout_ms: u64) -> Result<String, String> {
    let timeout = Duration::from_millis(timeout_ms.max(1));
    let address = (endpoint.host.as_str(), endpoint.port)
        .to_socket_addrs()
        .map_err(|err| format!("could not resolve NEOS endpoint {}: {err}", endpoint.url))?
        .next()
        .ok_or_else(|| format!("could not resolve NEOS endpoint {}", endpoint.url))?;
    let stream = TcpStream::connect_timeout(&address, timeout)
        .map_err(|err| format!("could not connect to NEOS endpoint {}: {err}", endpoint.url))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|err| format!("could not set NEOS read timeout: {err}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|err| format!("could not set NEOS write timeout: {err}"))?;

    let connector = TlsConnector::new()
        .map_err(|err| format!("could not initialize NEOS TLS connector: {err}"))?;
    let mut stream = connector.connect(&endpoint.host, stream).map_err(|err| {
        format!(
            "could not negotiate TLS with NEOS endpoint {}: {err}",
            endpoint.url
        )
    })?;
    let request = neos_ping_http_request(endpoint);
    stream
        .write_all(request.as_bytes())
        .map_err(|err| format!("could not write NEOS ping request: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("could not flush NEOS ping request: {err}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|err| format!("could not read NEOS ping response: {err}"))?;
    Ok(response)
}

fn neos_ping_http_request(endpoint: &NeosXmlRpcEndpoint) -> String {
    format!(
        "POST {} HTTP/1.1\r\n\
         Host: {}:{}\r\n\
         User-Agent: des-rs-neos-probe\r\n\
         Content-Type: text/xml\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        endpoint.path,
        endpoint.host,
        endpoint.port,
        NEOS_PING_XML.len(),
        NEOS_PING_XML
    )
}

fn classify_neos_ping_response(endpoint: &NeosXmlRpcEndpoint, response: &str) -> ExternalNeosProbe {
    let status_line = response.lines().next().unwrap_or_default();
    let http_ok = status_line.starts_with("HTTP/") && status_line.contains(" 200 ");
    let lower = response.to_ascii_lowercase();
    let compact = lower.split_whitespace().collect::<String>();
    let alive = compact.contains("neosserverisalive")
        || lower.contains("neos server is alive")
        || lower.contains("neosserver is alive");
    if http_ok && lower.contains("<methodresponse>") && alive {
        return ExternalNeosProbe {
            ready: true,
            endpoint: endpoint.url.clone(),
            message: format!(
                "NEOS XML-RPC ping succeeded at {} without submitting a job",
                endpoint.url
            ),
        };
    }
    ExternalNeosProbe {
        ready: false,
        endpoint: endpoint.url.clone(),
        message: format!(
            "NEOS XML-RPC ping at {} did not return the expected service response: {}",
            endpoint.url,
            short_response_excerpt(response)
        ),
    }
}

fn short_response_excerpt(response: &str) -> String {
    let normalized = response
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    const LIMIT: usize = 320;
    if normalized.chars().count() <= LIMIT {
        normalized
    } else {
        format!("{}...", normalized.chars().take(LIMIT).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_https_endpoint() {
        let endpoint = parse_neos_xmlrpc_endpoint(DEFAULT_NEOS_XMLRPC_URL).unwrap();
        assert_eq!(endpoint.host, "neos-server.org");
        assert_eq!(endpoint.port, 3333);
        assert_eq!(endpoint.path, "/");
    }

    #[test]
    fn rejects_non_https_endpoint() {
        let err = parse_neos_xmlrpc_endpoint("http://neos-server.org:3333").unwrap_err();
        assert!(err.contains("https://"), "{err}");
    }

    #[test]
    fn classifies_successful_ping_response() {
        let endpoint = parse_neos_xmlrpc_endpoint(DEFAULT_NEOS_XMLRPC_URL).unwrap();
        let response = "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\n\r\n<?xml version='1.0'?>\n<methodResponse>\n<params><param><value><string>NeosServer is alive\n</string></value></param></params>\n</methodResponse>";
        let probe = classify_neos_ping_response(&endpoint, response);
        assert!(probe.ready, "{}", probe.message);
    }

    #[test]
    fn rejects_unexpected_ping_response() {
        let endpoint = parse_neos_xmlrpc_endpoint(DEFAULT_NEOS_XMLRPC_URL).unwrap();
        let response = "HTTP/1.1 503 Service Unavailable\r\n\r\n<error>offline</error>";
        let probe = classify_neos_ping_response(&endpoint, response);
        assert!(!probe.ready);
        assert!(probe.message.contains("offline"), "{}", probe.message);
    }
}
