//! A catalogue of installed plugins, and the bridge that surfaces them in the
//! [`crate::des::service`] descriptor.
//!
//! [`PluginRegistry`] is a deduped (by `id`) collection of [`PluginManifest`]s.
//! [`PluginCatalogExtension`] adapts that catalogue to the [`DesExtension`]
//! seam so an embedding server gets, for free, one capability + a run/player
//! endpoint pair per plugin in its `/api/docs.json` descriptor — discoverable
//! exactly like the engine's built-in simulation catalogue.

use crate::des::service::{Capability, DesExtension, EndpointDoc, EndpointKind};

use super::manifest::PluginManifest;

/// Returned when registering a plugin whose id is already taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplicatePlugin(pub String);

impl std::fmt::Display for DuplicatePlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "plugin \"{}\" already registered", self.0)
    }
}

impl std::error::Error for DuplicatePlugin {}

/// An ordered, id-unique set of plugin manifests.
#[derive(Clone, Debug, Default)]
pub struct PluginRegistry {
    plugins: Vec<PluginManifest>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        PluginRegistry {
            plugins: Vec::new(),
        }
    }

    /// Register a plugin. Rejects a duplicate `id` (mirrors the service
    /// builder's duplicate-extension behaviour).
    pub fn register(&mut self, manifest: PluginManifest) -> Result<(), DuplicatePlugin> {
        if self.plugins.iter().any(|p| p.id == manifest.id) {
            return Err(DuplicatePlugin(manifest.id));
        }
        self.plugins.push(manifest);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&PluginManifest> {
        self.plugins.iter().find(|p| p.id == id)
    }

    pub fn list(&self) -> &[PluginManifest] {
        &self.plugins
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// The route a server mounts to run a plugin and stream its player HTML.
    pub fn run_route(id: &str) -> String {
        format!("/plugins/{id}/player")
    }

    /// All manifests as a JSON array (a machine-readable plugin catalogue).
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.plugins).unwrap_or_else(|_| "[]".to_string())
    }

    /// Adapt this catalogue to a [`DesExtension`] for the service descriptor.
    pub fn as_extension(&self) -> PluginCatalogExtension {
        PluginCatalogExtension {
            plugins: self.plugins.clone(),
        }
    }
}

/// [`DesExtension`] that advertises every plugin in a [`PluginRegistry`].
pub struct PluginCatalogExtension {
    plugins: Vec<PluginManifest>,
}

impl PluginCatalogExtension {
    pub const NAME: &'static str = "des-plugin-catalogue";
}

impl DesExtension for PluginCatalogExtension {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn endpoints(&self) -> Vec<EndpointDoc> {
        self.plugins
            .iter()
            .map(|p| {
                EndpointDoc::new(
                    "GET",
                    PluginRegistry::run_route(&p.id),
                    format!("Run plugin `{}` and render its player.", p.id),
                    EndpointKind::Action,
                )
            })
            .collect()
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.plugins
            .iter()
            .map(|p| Capability {
                name: p.id.clone(),
                description: if p.description.is_empty() {
                    format!("External plugin `{}`.", p.name)
                } else {
                    p.description.clone()
                },
                provided_by: Self::NAME.to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::plugin::manifest::{
        OutputKind, PlayerKind, PluginRuntimeKind, PluginTransportKind, RunSpec,
    };
    use crate::des::service::{ServiceBuilder, ServiceInfo};

    fn manifest(id: &str) -> PluginManifest {
        PluginManifest {
            id: id.to_string(),
            name: format!("Plugin {id}"),
            version: "1.0.0".to_string(),
            description: String::new(),
            runtime: PluginRuntimeKind::Rust,
            transport: PluginTransportKind::Stdio,
            language: None,
            run: RunSpec::new("./bin", &[]),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls: Vec::new(),
            title: None,
        }
    }

    #[test]
    fn registry_dedupes_by_id() {
        let mut r = PluginRegistry::new();
        r.register(manifest("a")).unwrap();
        assert_eq!(
            r.register(manifest("a")).unwrap_err(),
            DuplicatePlugin("a".to_string())
        );
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn extension_surfaces_plugins_in_descriptor() {
        let mut r = PluginRegistry::new();
        r.register(manifest("lp-stream")).unwrap();
        r.register(manifest("mm1")).unwrap();

        let mut b = ServiceBuilder::new(ServiceInfo {
            name: "svc".to_string(),
            version: "0.1.0".to_string(),
            description: "test".to_string(),
        });
        b.register(Box::new(r.as_extension())).unwrap();
        let d = b.build();

        assert!(d.capabilities.iter().any(|c| c.name == "lp-stream"));
        assert!(d.capabilities.iter().any(|c| c.name == "mm1"));
        assert!(d.endpoints.iter().any(|e| e.path == "/plugins/mm1/player"));
        let ep = d
            .endpoints
            .iter()
            .find(|e| e.path == "/plugins/mm1/player")
            .unwrap();
        assert_eq!(
            ep.provided_by.as_deref(),
            Some(PluginCatalogExtension::NAME)
        );
    }
}
