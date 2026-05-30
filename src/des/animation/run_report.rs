//! Port of `src/des/animation/run-report.ts`.
//!
//! Class-only HTML builders for non-animation simulation runs:
//!
//! * [`RunReportPage`] — a styled single-run report (header, metric tables,
//!   captured console output).
//! * [`SimulationIndexPage`] — the curated landing page listing every
//!   simulation/run and linking to its HTML.
//!
//! Pages use relative links and a dark theme matching the animation player.
//!
//! ## Conversion notes
//!
//! * `static escape(input)` → the associated fn [`RunReportPage::escape`]
//!   (chained `String.replace` → chained [`str::replace`]).
//! * Builder methods returning `this` → return `&mut Self` for chaining.
//! * Backtick template literals → `format!` with the CSS held in a separate
//!   `const &str` (so its `{}` are data, never format syntax); the markup is
//!   reproduced byte-for-byte.

#![allow(dead_code)]

// =============================================================================
// RunReportPage.
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct MetricRow {
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug, Default)]
pub struct ReportSection {
    pub heading: String,
    pub description: Option<String>,
    pub metrics: Option<Vec<MetricRow>>,
    /// Monospaced block (e.g. captured stdout). Rendered verbatim.
    pub log: Option<String>,
}

pub struct RunReportPage {
    title: String,
    subtitle: String,
    back_href: String,
    sections: Vec<ReportSection>,
}

impl RunReportPage {
    /// `new RunReportPage(title, subtitle)` — `backHref` defaults to
    /// `'../index.html'`.
    pub fn new(title: impl Into<String>, subtitle: impl Into<String>) -> Self {
        Self::with_back_href(title, subtitle, "../index.html")
    }

    /// `new RunReportPage(title, subtitle, backHref)`.
    pub fn with_back_href(
        title: impl Into<String>,
        subtitle: impl Into<String>,
        back_href: impl Into<String>,
    ) -> Self {
        RunReportPage {
            title: title.into(),
            subtitle: subtitle.into(),
            back_href: back_href.into(),
            sections: Vec::new(),
        }
    }

    pub fn add_section(&mut self, section: ReportSection) -> &mut Self {
        self.sections.push(section);
        self
    }

    /// HTML entity escaping (`& < > "` — note: no `'`, matching the TS).
    pub fn escape(input: &str) -> String {
        input
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    fn render_section(&self, s: &ReportSection) -> String {
        let mut parts: Vec<String> =
            vec![format!("<section><h2>{}</h2>", Self::escape(&s.heading))];
        if let Some(desc) = &s.description {
            if !desc.is_empty() {
                parts.push(format!("<p class=\"desc\">{}</p>", Self::escape(desc)));
            }
        }
        if let Some(metrics) = &s.metrics {
            if !metrics.is_empty() {
                parts.push("<table><tbody>".to_string());
                for m in metrics {
                    parts.push(format!(
                        "<tr><th>{}</th><td>{}</td></tr>",
                        Self::escape(&m.label),
                        Self::escape(&m.value)
                    ));
                }
                parts.push("</tbody></table>".to_string());
            }
        }
        if let Some(log) = &s.log {
            if !log.is_empty() {
                parts.push(format!("<pre class=\"log\">{}</pre>", Self::escape(log)));
            }
        }
        parts.push("</section>".to_string());
        parts.join("")
    }

    pub fn to_html(&self) -> String {
        let body = self
            .sections
            .iter()
            .map(|s| self.render_section(s))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style></head><body><main>
<a class="back" href="{back}">&larr; all simulations</a>
<h1>{title}</h1>
<p class="sub">{sub}</p>
{body}
</main></body></html>"#,
            title = Self::escape(&self.title),
            css = CSS_REPORT,
            back = Self::escape(&self.back_href),
            sub = Self::escape(&self.subtitle),
            body = body,
        )
    }
}

// =============================================================================
// SimulationIndexPage.
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct IndexEntry {
    pub title: String,
    pub description: String,
    pub href: String,
    /// Short tag shown on the card, e.g. "animation" or "run report".
    pub kind: String,
}

#[derive(Clone, Debug, Default)]
pub struct IndexGroup {
    pub heading: String,
    pub blurb: String,
    pub entries: Vec<IndexEntry>,
}

/// A compact link in the full directory catalog.
#[derive(Clone, Debug, Default)]
pub struct CatalogEntry {
    pub href: String,
    pub label: String,
    pub size: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CatalogSection {
    pub heading: String,
    pub blurb: String,
    pub entries: Vec<CatalogEntry>,
}

pub struct SimulationIndexPage {
    title: String,
    subtitle: String,
    groups: Vec<IndexGroup>,
    catalogs: Vec<CatalogSection>,
}

impl SimulationIndexPage {
    pub fn new(title: impl Into<String>, subtitle: impl Into<String>) -> Self {
        SimulationIndexPage {
            title: title.into(),
            subtitle: subtitle.into(),
            groups: Vec::new(),
            catalogs: Vec::new(),
        }
    }

    pub fn add_group(&mut self, group: IndexGroup) -> &mut Self {
        self.groups.push(group);
        self
    }

    pub fn add_catalog(&mut self, section: CatalogSection) -> &mut Self {
        self.catalogs.push(section);
        self
    }

    fn render_entry(&self, e: &IndexEntry) -> String {
        let esc = RunReportPage::escape;
        format!(
            r#"<a class="card" href="{href}">
<span class="tag">{kind}</span>
<span class="card-title">{title}</span>
<span class="card-desc">{desc}</span>
</a>"#,
            href = esc(&e.href),
            kind = esc(&e.kind),
            title = esc(&e.title),
            desc = esc(&e.description),
        )
    }

    fn render_group(&self, g: &IndexGroup) -> String {
        let esc = RunReportPage::escape;
        let entries = g
            .entries
            .iter()
            .map(|e| self.render_entry(e))
            .collect::<Vec<_>>()
            .join("");
        format!(
            r#"<section><h2>{heading}</h2><p class="blurb">{blurb}</p>
<div class="grid">{entries}</div></section>"#,
            heading = esc(&g.heading),
            blurb = esc(&g.blurb),
            entries = entries,
        )
    }

    fn render_catalog(&self, c: &CatalogSection) -> String {
        let esc = RunReportPage::escape;
        let rows = c
            .entries
            .iter()
            .map(|e| {
                let size = match &e.size {
                    Some(s) if !s.is_empty() => format!("<span class=\"size\">{}</span>", esc(s)),
                    _ => String::new(),
                };
                format!(
                    "<li><a href=\"{}\"><span class=\"path\">{}</span>{}</a></li>",
                    esc(&e.href),
                    esc(&e.label),
                    size
                )
            })
            .collect::<Vec<_>>()
            .join("");
        format!(
            r#"<section><h2>{heading} <span class="count">{count}</span></h2>
<p class="blurb">{blurb}</p><ul class="catalog">{rows}</ul></section>"#,
            heading = esc(&c.heading),
            count = c.entries.len(),
            blurb = esc(&c.blurb),
            rows = rows,
        )
    }

    pub fn to_html(&self, generated_at: &str) -> String {
        let esc = RunReportPage::escape;
        let groups = self
            .groups
            .iter()
            .map(|g| self.render_group(g))
            .collect::<Vec<_>>()
            .join("\n");
        let catalogs = self
            .catalogs
            .iter()
            .map(|c| self.render_catalog(c))
            .collect::<Vec<_>>()
            .join("\n");
        let body = format!("{groups}\n{catalogs}");
        format!(
            r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style></head><body><main>
<h1>{title}</h1>
<p class="sub">{sub}</p>
{body}
<footer>Generated {gen} · served from the discrete-event-system <code>out/</code> directory.</footer>
</main></body></html>"#,
            title = esc(&self.title),
            css = CSS_INDEX,
            sub = esc(&self.subtitle),
            body = body,
            gen = esc(generated_at),
        )
    }
}

// -----------------------------------------------------------------------------
// Stylesheets (held separately so the `{...}` rules are never format syntax).
// -----------------------------------------------------------------------------

const CSS_REPORT: &str = r#":root{color-scheme:dark;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0b1021;color:#e6edf3;}
main{max-width:960px;margin:0 auto;padding:28px 20px 72px;}
a.back{color:#58a6ff;text-decoration:none;font-size:.9rem;}
a.back:hover{text-decoration:underline;}
h1{font-size:1.7rem;margin:14px 0 4px;}
p.sub{color:#8b949e;margin:0 0 26px;font-size:.95rem;}
section{background:#161d33;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin:0 0 20px;}
h2{font-size:1.15rem;margin:0 0 8px;color:#f0f6fc;}
p.desc{color:#9aa5b1;margin:0 0 14px;font-size:.92rem;}
table{border-collapse:collapse;width:100%;margin:0 0 4px;}
th{text-align:left;color:#8b949e;font-weight:600;font-size:.82rem;padding:6px 12px 6px 0;white-space:nowrap;vertical-align:top;}
td{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86rem;color:#e6edf3;padding:6px 0;}
pre.log{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 16px;overflow:auto;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;line-height:1.5;color:#c9d1d9;}"#;

const CSS_INDEX: &str = r#":root{color-scheme:dark;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0b1021;color:#e6edf3;}
main{max-width:1040px;margin:0 auto;padding:36px 22px 80px;}
h1{font-size:2rem;margin:0 0 6px;}
p.sub{color:#8b949e;margin:0 0 32px;font-size:1rem;}
section{margin:0 0 34px;}
h2{font-size:1.25rem;margin:0 0 4px;color:#f0f6fc;}
p.blurb{color:#9aa5b1;margin:0 0 16px;font-size:.92rem;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;}
a.card{display:flex;flex-direction:column;gap:6px;background:#161d33;border:1px solid #21262d;
border-radius:12px;padding:16px 18px;text-decoration:none;transition:border-color .15s,transform .15s;}
a.card:hover{border-color:#58a6ff;transform:translateY(-2px);}
.tag{align-self:flex-start;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:#0b1021;
background:#58a6ff;border-radius:999px;padding:2px 9px;font-weight:700;}
.card-title{color:#f0f6fc;font-size:1.05rem;font-weight:600;}
.card-desc{color:#9aa5b1;font-size:.86rem;line-height:1.4;}
.count{display:inline-block;font-size:.72rem;color:#0b1021;background:#7d8590;border-radius:999px;
padding:1px 8px;vertical-align:middle;font-weight:700;}
ul.catalog{list-style:none;padding:0;margin:0;column-width:330px;column-gap:14px;}
ul.catalog li{break-inside:avoid;margin:0 0 4px;}
ul.catalog a{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
padding:6px 10px;border:1px solid #21262d;border-radius:7px;text-decoration:none;background:#11172b;}
ul.catalog a:hover{border-color:#58a6ff;background:#161d33;}
ul.catalog .path{color:#58a6ff;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
ul.catalog .size{color:#586069;font-size:.72rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;}
footer{color:#586069;font-size:.8rem;margin-top:20px;}
footer code{color:#8b949e;}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_handles_the_four_entities() {
        assert_eq!(
            RunReportPage::escape("a&b<c>\"d"),
            "a&amp;b&lt;c&gt;&quot;d"
        );
        // Single quotes are intentionally NOT escaped.
        assert_eq!(RunReportPage::escape("it's"), "it's");
    }

    #[test]
    fn report_page_renders_sections() {
        let mut page = RunReportPage::new("My <Run>", "a subtitle");
        page.add_section(ReportSection {
            heading: "Metrics".to_string(),
            description: Some("desc".to_string()),
            metrics: Some(vec![MetricRow {
                label: "n".to_string(),
                value: "42".to_string(),
            }]),
            log: Some("hello".to_string()),
        });
        let html = page.to_html();
        assert!(html.contains("<title>My &lt;Run&gt;</title>"));
        assert!(html.contains("<h1>My &lt;Run&gt;</h1>"));
        assert!(html.contains("<a class=\"back\" href=\"../index.html\">"));
        assert!(html.contains("<tr><th>n</th><td>42</td></tr>"));
        assert!(html.contains("<pre class=\"log\">hello</pre>"));
        assert!(html.trim_end().ends_with("</html>"));
    }

    #[test]
    fn index_page_renders_groups_and_catalogs() {
        let mut page = SimulationIndexPage::new("Index", "sub");
        page.add_group(IndexGroup {
            heading: "Group".to_string(),
            blurb: "blurb".to_string(),
            entries: vec![IndexEntry {
                title: "Sim".to_string(),
                description: "does things".to_string(),
                href: "sim.html".to_string(),
                kind: "animation".to_string(),
            }],
        });
        page.add_catalog(CatalogSection {
            heading: "All".to_string(),
            blurb: "everything".to_string(),
            entries: vec![CatalogEntry {
                href: "a.html".to_string(),
                label: "a".to_string(),
                size: Some("1kb".to_string()),
            }],
        });
        let html = page.to_html("2026-01-01");
        assert!(html.contains("<span class=\"tag\">animation</span>"));
        assert!(html.contains("<span class=\"count\">1</span>"));
        assert!(html.contains("<span class=\"size\">1kb</span>"));
        assert!(html.contains("Generated 2026-01-01 ·"));
    }
}
