//! Service self-description, a plugin/extension system, and machine discovery
//! for any HTTP server that embeds this engine as a library.
//!
//! The engine is consumed by several servers (today the `dd-des-rs` axum
//! service; tomorrow others). Rather than have each server hand-maintain its
//! own route inventory and reinvent a discovery convention, that contract lives
//! here so every consumer self-describes identically.
//!
//! ## JSON-first
//!
//! This module is deliberately **JSON-first**: the canonical, machine-readable
//! artifact is the [`ServiceDescriptor`] and its serialization
//! ([`ServiceDescriptor::to_json_string`]). The library does NOT render HTML —
//! presentation is a consumer concern. Each embedding server renders its own
//! (branded) docs page as a *view* over this descriptor, so there is exactly
//! one source of truth (the JSON) and many independent renderings.
//!
//! ## Plugins / extensions
//!
//! [`DesExtension`] is the plugin seam. An extension contributes endpoints
//! and/or capabilities; servers (or future engine plugins) register them on a
//! [`ServiceBuilder`] without touching the core. The builder dedupes by name
//! the same way [`crate::des::general::des_registry::Registry`] dedupes models,
//! returning a typed [`ServiceError`] for the one recoverable failure (a
//! duplicate extension).
//!
//! ## Discovery
//!
//! [`ServiceDescriptor::link_header_relative`] builds an RFC 8288 `Link` header
//! (relations from RFC 8631 — `service-doc` for the human docs page,
//! `service-desc` for this machine descriptor). A crawler that hits only the
//! canonical landing route learns where the docs live; resolving the relative
//! targets against the request URL works whether the service is mounted at `/`
//! or behind a path-stripping gateway at `/<prefix>/`. A first-party
//! [`DD_API_DOCS_HEADER`] is emitted alongside for dd tooling.
//!
//! `serde` is used only here, at the HTTP-presentation boundary — never in the
//! engine core (see the crate's Rust-migration conventions).

use serde::Serialize;

// =============================================================================
// Canonical route + header constants (shared by every embedding server).
// =============================================================================

/// Canonical human-readable HTML docs route every embedding server mounts.
pub const DOCS_HTML_ROUTE: &str = "/docs/api";
/// Alias of [`DOCS_HTML_ROUTE`] (the cluster convention mounts both).
pub const DOCS_HTML_ROUTE_ALT: &str = "/api/docs";
/// Canonical machine-readable JSON docs route (serves [`ServiceDescriptor`]).
pub const DOCS_JSON_ROUTE: &str = "/api/docs.json";

/// RFC 8631 link relation for human-readable service documentation.
pub const REL_SERVICE_DOC: &str = "service-doc";
/// RFC 8631 link relation for a machine-readable service description.
pub const REL_SERVICE_DESC: &str = "service-desc";

/// First-party discovery header naming the machine-readable docs location.
pub const DD_API_DOCS_HEADER: &str = "dd-server-api-docs";

// =============================================================================
// Descriptor value types (the JSON contract).
// =============================================================================

/// Coarse classification of an endpoint, so docs/discovery tooling can group
/// routes without parsing paths. Serializes to a lowercase-kebab string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EndpointKind {
    /// Operational / infrastructure routes (health, landing, static artifacts).
    Service,
    /// The self-describing docs routes themselves.
    Docs,
    /// A domain action the service performs (e.g. running a simulation).
    Action,
    /// Anything else a server or plugin defines.
    Custom,
}

/// One HTTP endpoint in a [`ServiceDescriptor`].
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EndpointDoc {
    pub method: String,
    pub path: String,
    pub description: String,
    pub kind: EndpointKind,
    /// Name of the extension that contributed this endpoint, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provided_by: Option<String>,
}

impl EndpointDoc {
    pub fn new(
        method: impl Into<String>,
        path: impl Into<String>,
        description: impl Into<String>,
        kind: EndpointKind,
    ) -> Self {
        EndpointDoc {
            method: method.into(),
            path: path.into(),
            description: description.into(),
            kind,
            provided_by: None,
        }
    }
}

/// A non-route capability the service exposes (e.g. a runnable simulation),
/// surfaced so machines can enumerate what the service can do.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capability {
    pub name: String,
    pub description: String,
    pub provided_by: String,
}

/// Identity + summary of the service being described.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub description: String,
}

/// Summary of a registered extension, echoed into the descriptor.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionInfo {
    pub name: String,
    pub version: String,
    pub endpoint_count: usize,
    pub capability_count: usize,
}

/// The canonical doc-route paths, echoed in the descriptor so a machine that
/// has the JSON does not have to know the conventional paths out of band.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DocsRoutes {
    pub html: String,
    pub html_alt: String,
    pub json: String,
}

impl Default for DocsRoutes {
    fn default() -> Self {
        DocsRoutes {
            html: DOCS_HTML_ROUTE.to_string(),
            html_alt: DOCS_HTML_ROUTE_ALT.to_string(),
            json: DOCS_JSON_ROUTE.to_string(),
        }
    }
}

/// Full, machine-readable description of a service's HTTP surface. This is the
/// single source of truth; HTML docs pages are rendered independently from it.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceDescriptor {
    /// Descriptor format version, so consumers can evolve safely.
    pub schema: String,
    #[serde(flatten)]
    pub info: ServiceInfo,
    pub docs: DocsRoutes,
    pub endpoints: Vec<EndpointDoc>,
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<ExtensionInfo>,
}

/// The value of [`ServiceDescriptor::schema`].
pub const SERVICE_DESCRIPTOR_SCHEMA: &str = "des/service-descriptor/v1";

impl ServiceDescriptor {
    /// Pretty-printed canonical JSON (what `/api/docs.json` serves). Uses the
    /// crate's `serde_json` boundary dependency, so even servers without a
    /// serde-aware response type can serve the descriptor verbatim.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// RFC 8288 `Link` header value with RELATIVE targets, so it resolves
    /// against whatever URL the client used — `/` locally, `/<prefix>/` behind
    /// a path-stripping gateway. Emit this on the canonical landing route: a
    /// request to `/<prefix>/` resolves `docs/api` to `/<prefix>/docs/api`.
    pub fn link_header_relative(&self) -> String {
        format!(
            "<{html}>; rel=\"{doc}\"; type=\"text/html\", <{json}>; rel=\"{desc}\"; type=\"application/json\"",
            html = strip_leading_slash(&self.docs.html),
            json = strip_leading_slash(&self.docs.json),
            doc = REL_SERVICE_DOC,
            desc = REL_SERVICE_DESC,
        )
    }

    /// RFC 8288 `Link` header value with ABSOLUTE targets under a known mount
    /// `prefix` (e.g. `"/des-rs"`, or `""` for root). Use when the server or
    /// gateway knows the public prefix and wants correct discovery headers on
    /// every response, not just the landing route.
    pub fn link_header_at(&self, prefix: &str) -> String {
        let p = prefix.trim_end_matches('/');
        format!(
            "<{p}{html}>; rel=\"{doc}\"; type=\"text/html\", <{p}{json}>; rel=\"{desc}\"; type=\"application/json\"",
            html = self.docs.html,
            json = self.docs.json,
            doc = REL_SERVICE_DOC,
            desc = REL_SERVICE_DESC,
        )
    }

    /// Relative value for the [`DD_API_DOCS_HEADER`] discovery header (the
    /// machine descriptor location).
    pub fn dd_api_docs_relative(&self) -> String {
        strip_leading_slash(&self.docs.json).to_string()
    }

    /// Count of endpoints with the given [`EndpointKind`] (handy for renderers).
    pub fn endpoint_count(&self, kind: EndpointKind) -> usize {
        self.endpoints.iter().filter(|e| e.kind == kind).count()
    }
}

fn strip_leading_slash(s: &str) -> &str {
    s.strip_prefix('/').unwrap_or(s)
}

// =============================================================================
// Plugin / extension seam.
// =============================================================================

/// The plugin seam: an extension contributes endpoints and/or capabilities to a
/// service. `Send + Sync` so servers can register engine plugins and then share
/// the built descriptor across async worker tasks.
pub trait DesExtension: Send + Sync {
    /// Unique extension name (used for dedupe and attribution).
    fn name(&self) -> &str;
    /// Extension version; defaults to unversioned.
    fn version(&self) -> &str {
        "0.0.0"
    }
    /// HTTP endpoints this extension adds. Default: none.
    fn endpoints(&self) -> Vec<EndpointDoc> {
        Vec::new()
    }
    /// Capabilities this extension adds. Default: none.
    fn capabilities(&self) -> Vec<Capability> {
        Vec::new()
    }
}

/// The one recoverable failure while building a descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceError {
    DuplicateExtension(String),
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::DuplicateExtension(name) => {
                write!(f, "extension \"{name}\" already registered")
            }
        }
    }
}

impl std::error::Error for ServiceError {}

/// Collects a service's own (core) endpoints plus registered [`DesExtension`]s
/// and produces a [`ServiceDescriptor`]. Mirrors the dedupe-by-name behaviour of
/// [`crate::des::general::des_registry::Registry`].
pub struct ServiceBuilder {
    info: ServiceInfo,
    core_endpoints: Vec<EndpointDoc>,
    extensions: Vec<Box<dyn DesExtension>>,
    extension_names: Vec<String>,
}

impl ServiceBuilder {
    pub fn new(info: ServiceInfo) -> Self {
        ServiceBuilder {
            info,
            core_endpoints: Vec::new(),
            extensions: Vec::new(),
            extension_names: Vec::new(),
        }
    }

    /// Add one of the service's own endpoints (not contributed by a plugin).
    pub fn endpoint(
        &mut self,
        method: impl Into<String>,
        path: impl Into<String>,
        description: impl Into<String>,
        kind: EndpointKind,
    ) -> &mut Self {
        self.core_endpoints
            .push(EndpointDoc::new(method, path, description, kind));
        self
    }

    /// Register an extension. Duplicate names are rejected (the `des_registry`
    /// `AlreadyRegistered` analogue).
    pub fn register(&mut self, ext: Box<dyn DesExtension>) -> Result<(), ServiceError> {
        let name = ext.name().to_string();
        if self.extension_names.iter().any(|n| n == &name) {
            return Err(ServiceError::DuplicateExtension(name));
        }
        self.extension_names.push(name);
        self.extensions.push(ext);
        Ok(())
    }

    /// Aggregate core endpoints, every extension's contributions, and the
    /// canonical docs routes into a [`ServiceDescriptor`].
    pub fn build(&self) -> ServiceDescriptor {
        let mut endpoints: Vec<EndpointDoc> = self.core_endpoints.clone();
        let mut capabilities: Vec<Capability> = Vec::new();
        let mut extension_infos: Vec<ExtensionInfo> = Vec::new();

        for ext in &self.extensions {
            let ep = ext.endpoints();
            let cap = ext.capabilities();
            extension_infos.push(ExtensionInfo {
                name: ext.name().to_string(),
                version: ext.version().to_string(),
                endpoint_count: ep.len(),
                capability_count: cap.len(),
            });
            for mut e in ep {
                if e.provided_by.is_none() {
                    e.provided_by = Some(ext.name().to_string());
                }
                endpoints.push(e);
            }
            capabilities.extend(cap);
        }

        // Always advertise the canonical docs routes, deduped against anything
        // a server/plugin already declared. The descriptor lists them; the
        // server is still responsible for actually mounting them.
        for (method, path, desc) in standard_doc_endpoints() {
            if !endpoints
                .iter()
                .any(|e| e.path == path && e.method == method)
            {
                endpoints.push(EndpointDoc::new(method, path, desc, EndpointKind::Docs));
            }
        }

        ServiceDescriptor {
            schema: SERVICE_DESCRIPTOR_SCHEMA.to_string(),
            info: self.info.clone(),
            docs: DocsRoutes::default(),
            endpoints,
            capabilities,
            extensions: extension_infos,
        }
    }
}

fn standard_doc_endpoints() -> [(&'static str, &'static str, &'static str); 3] {
    [
        (
            "GET",
            DOCS_HTML_ROUTE,
            "Human-readable HTML API documentation (a view rendered from the JSON descriptor).",
        ),
        ("GET", DOCS_HTML_ROUTE_ALT, "Alias of /docs/api."),
        (
            "GET",
            DOCS_JSON_ROUTE,
            "Machine-readable API descriptor (this document).",
        ),
    ]
}

// =============================================================================
// Built-in extension: the engine's simulation catalogue.
// =============================================================================

/// Built-in extension that surfaces the engine's simulation catalogue as
/// capabilities, so every embedding server advertises the same runnable set
/// without re-listing it.
pub struct EngineCatalogExtension;

impl DesExtension for EngineCatalogExtension {
    fn name(&self) -> &str {
        "des-engine-catalogue"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn capabilities(&self) -> Vec<Capability> {
        crate::des::simulations::simulation_catalogue()
            .into_iter()
            .map(|(name, _)| Capability {
                name: name.to_string(),
                description: format!("Runnable discrete-event simulation `{name}`."),
                provided_by: "des-engine-catalogue".to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> ServiceInfo {
        ServiceInfo {
            name: "test-svc".to_string(),
            version: "1.2.3".to_string(),
            description: "a test service".to_string(),
        }
    }

    struct FakeExt;
    impl DesExtension for FakeExt {
        fn name(&self) -> &str {
            "fake"
        }
        fn endpoints(&self) -> Vec<EndpointDoc> {
            vec![EndpointDoc::new(
                "POST",
                "/fake",
                "fake action",
                EndpointKind::Action,
            )]
        }
        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability {
                name: "fake-cap".to_string(),
                description: "a fake capability".to_string(),
                provided_by: "fake".to_string(),
            }]
        }
    }

    #[test]
    fn build_includes_core_extension_and_standard_doc_routes() {
        let mut b = ServiceBuilder::new(info());
        b.endpoint("GET", "/", "landing", EndpointKind::Service);
        b.register(Box::new(FakeExt)).unwrap();
        let d = b.build();

        assert_eq!(d.schema, SERVICE_DESCRIPTOR_SCHEMA);
        assert_eq!(d.info.name, "test-svc");
        // core + the 3 standard doc routes + the fake extension endpoint.
        assert!(d.endpoints.iter().any(|e| e.path == "/"));
        assert!(d.endpoints.iter().any(|e| e.path == DOCS_HTML_ROUTE));
        assert!(d.endpoints.iter().any(|e| e.path == DOCS_HTML_ROUTE_ALT));
        assert!(d.endpoints.iter().any(|e| e.path == DOCS_JSON_ROUTE));
        // the extension endpoint is attributed to its plugin.
        let fake = d.endpoints.iter().find(|e| e.path == "/fake").unwrap();
        assert_eq!(fake.provided_by.as_deref(), Some("fake"));
        assert!(d.capabilities.iter().any(|c| c.name == "fake-cap"));
        assert_eq!(d.extensions.len(), 1);
        assert_eq!(d.extensions[0].endpoint_count, 1);
    }

    #[test]
    fn duplicate_extension_is_rejected() {
        let mut b = ServiceBuilder::new(info());
        b.register(Box::new(FakeExt)).unwrap();
        let err = b.register(Box::new(FakeExt)).unwrap_err();
        assert_eq!(err, ServiceError::DuplicateExtension("fake".to_string()));
    }

    #[test]
    fn link_header_is_relative_and_carries_both_relations() {
        let d = ServiceBuilder::new(info()).build();
        let link = d.link_header_relative();
        // relative targets (no leading slash) so they resolve under any mount.
        assert!(link.contains("<docs/api>"));
        assert!(link.contains("<api/docs.json>"));
        assert!(link.contains(&format!("rel=\"{REL_SERVICE_DOC}\"")));
        assert!(link.contains(&format!("rel=\"{REL_SERVICE_DESC}\"")));
        assert_eq!(d.dd_api_docs_relative(), "api/docs.json");
    }

    #[test]
    fn link_header_at_prefix_is_absolute() {
        let d = ServiceBuilder::new(info()).build();
        let link = d.link_header_at("/des-rs/");
        assert!(link.contains("</des-rs/docs/api>"));
        assert!(link.contains("</des-rs/api/docs.json>"));
    }

    #[test]
    fn engine_catalogue_extension_lists_simulations() {
        let mut b = ServiceBuilder::new(info());
        b.register(Box::new(EngineCatalogExtension)).unwrap();
        let d = b.build();
        assert!(
            d.capabilities.len() >= 56,
            "expected the engine catalogue to be surfaced as capabilities"
        );
        assert!(d.capabilities.iter().any(|c| c.name == "main_build_site"));
    }

    #[test]
    fn json_is_valid_and_contains_contract_fields() {
        let mut b = ServiceBuilder::new(info());
        b.register(Box::new(EngineCatalogExtension)).unwrap();
        let json = b.build().to_json_string();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["schema"], SERVICE_DESCRIPTOR_SCHEMA);
        assert_eq!(parsed["name"], "test-svc");
        assert!(parsed["endpoints"].is_array());
        assert!(parsed["capabilities"].is_array());
        assert!(parsed["docs"]["json"] == DOCS_JSON_ROUTE);
    }
}
