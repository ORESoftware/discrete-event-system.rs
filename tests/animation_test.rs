//! Integration-test migration scaffold generated from the TypeScript test.
//! TypeScript source: `src/des/test/animation-test.ts`
//! Rust target: `tests/animation_test.rs`

#[test]
fn build_html_produces_self_contained_player() -> anyhow::Result<()> {
    use discrete_event_system_rs::des::animation::html_player::build_html;
    use discrete_event_system_rs::des::animation::types::{
        Animation, ChartSeries, ChartSpec, CircleShape, Frame, RectShape, Shape, TextAnchor,
        TextShape,
    };

    let anim = Animation {
        width: 400.0,
        height: 300.0,
        fps: 30.0,
        title: Some("Demo <ok>".to_owned()),
        subtitle: Some("sub".to_owned()),
        frames: vec![Frame {
            t: 0.0,
            tick: 0,
            shapes: vec![
                Shape::Circle(CircleShape {
                    x: 50.0,
                    y: 150.0,
                    r: 10.0,
                    fill: "#f00".to_owned(),
                    stroke: None,
                    stroke_width: None,
                    opacity: None,
                    label: Some("A".to_owned()),
                    title: None,
                    visual_block_id: None,
                }),
                Shape::Rect(RectShape {
                    x: 0.0,
                    y: 280.0,
                    w: 400.0,
                    h: 20.0,
                    fill: "#eee".to_owned(),
                    stroke: Some("#ccc".to_owned()),
                    stroke_width: None,
                    opacity: None,
                    label: None,
                    rx: None,
                    title: None,
                    visual_block_id: None,
                }),
                Shape::Text(TextShape {
                    x: 200.0,
                    y: 30.0,
                    text: "tick </script><script>alert(1)</script>".to_owned(),
                    font_size: Some(14.0),
                    fill: None,
                    anchor: Some(TextAnchor::Middle),
                    font_weight: None,
                    font_family: None,
                    visual_block_id: None,
                }),
            ],
            caption: Some("frame at t=0.00".to_owned()),
        }],
        charts: Some(vec![ChartSpec {
            x: 0.0,
            y: 200.0,
            w: 400.0,
            h: 80.0,
            title: Some("sample".to_owned()),
            y_min: None,
            y_max: None,
            y_label: None,
            series: vec![ChartSeries {
                label: "sin".to_owned(),
                color: "#08f".to_owned(),
                t: vec![0.0, 0.1],
                y: vec![0.0, 0.3],
            }],
            cursor: None,
        }]),
        background: None,
    };

    let json = serde_json::to_string(&anim)?;
    assert!(json.contains(r#""kind":"circle""#));
    assert!(json.contains(r##""stroke":"#ccc""##));
    assert!(json.contains(r#""fontSize":14.0"#));
    assert!(json.contains(r#""anchor":"middle""#));

    let html = build_html(&anim)?;
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains(r#"id="anim-data""#));
    assert!(html.contains(r#"id="stage""#));
    assert!(html.contains(r#"id="play""#));
    assert!(html.contains(r#"id="scrub""#));
    assert!(html.contains("function renderShape"));
    assert!(html.contains("function renderChart"));
    assert!(html.contains("requestAnimationFrame"));
    assert!(html.contains("Demo &lt;ok&gt;"));
    assert!(html.contains(r#"<\/script><script>alert(1)<\/script>"#));

    Ok(())
}

#[test]
fn build_html_set_embeds_variants_and_rejects_empty_sets() -> anyhow::Result<()> {
    use discrete_event_system_rs::des::animation::html_player::{
        build_html_set, AnimationSetOptions, AnimationVariant, HtmlRenderError,
    };
    use discrete_event_system_rs::des::animation::types::{Animation, Frame};

    let anim = Animation {
        width: 100.0,
        height: 100.0,
        fps: 24.0,
        title: Some("Variant demo".to_owned()),
        subtitle: None,
        frames: vec![Frame {
            t: 0.0,
            tick: 0,
            shapes: vec![],
            caption: None,
        }],
        charts: None,
        background: Some("#fff".to_owned()),
    };

    let variant = AnimationVariant {
        id: "base".to_owned(),
        label: "Base".to_owned(),
        animation: anim,
        summary: Some("smoke".to_owned()),
        controls: None,
    };
    let html = build_html_set(
        &[variant],
        AnimationSetOptions {
            title: Some("Set".to_owned()),
            subtitle: None,
            selector_label: Some("scenario".to_owned()),
        },
    )?;

    assert!(html.contains(r#""variants":[{"id":"base""#));
    assert!(html.contains(r#""selectorLabel":"scenario""#));
    assert!(html.contains("Variant demo") || html.contains("Set"));

    let err = build_html_set(&[], AnimationSetOptions::default()).unwrap_err();
    assert!(matches!(err, HtmlRenderError::EmptyAnimationSet));

    Ok(())
}
