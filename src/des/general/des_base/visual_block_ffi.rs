//! C ABI views over the renderer-neutral visual block IR.
//!
//! The canonical visual-block contract remains the Rust [`VisualBlockSpec`] /
//! JSON IR. This module adds an embedder-facing pointer mode for hosts such as
//! Unity, C++, C#, or native immediate-mode UIs that want to walk block, port,
//! and connection arrays directly instead of parsing JSON on every frame.
//!
//! All pointers returned from this module are read-only and remain valid until
//! the owning [`DesVisualBlockIrHandle`] is freed with
//! [`des_visual_block_ir_free`].

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::des::general::des_base::visual_block::{
    visual_block_ir, visual_block_specs, ResolvedLayout, VisualBlock, VisualBlockConnectionOptions,
    VisualBlockOptions, VisualBlockPort, VisualBlockSpec, VisualBlockStyle, VisualPortInput,
    VisualPortOptions,
};

const VISUAL_BLOCK_IR_SCHEMA_C: &[u8] = b"des/visual-block-ir/v1\0";

/// FFI export configuration.
///
/// Flags use `u8` rather than Rust `bool` so C, C++, C#, and Unity P/Invoke can
/// pass the struct by value without relying on language-specific bool layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesVisualBlockFfiConfig {
    /// Include a nul-terminated JSON representation of the same IR.
    pub include_json: u8,
    /// Include the top-level block slice.
    pub include_blocks: u8,
    /// Include port slices referenced by each block.
    pub include_ports: u8,
    /// Include the top-level connection slice.
    pub include_connections: u8,
}

impl DesVisualBlockFfiConfig {
    pub const fn all_views() -> Self {
        DesVisualBlockFfiConfig {
            include_json: 1,
            include_blocks: 1,
            include_ports: 1,
            include_connections: 1,
        }
    }

    pub const fn pointer_views() -> Self {
        DesVisualBlockFfiConfig {
            include_json: 0,
            include_blocks: 1,
            include_ports: 1,
            include_connections: 1,
        }
    }

    pub const fn json_only() -> Self {
        DesVisualBlockFfiConfig {
            include_json: 1,
            include_blocks: 0,
            include_ports: 0,
            include_connections: 0,
        }
    }

    fn wants_json(self) -> bool {
        self.include_json != 0
    }

    fn wants_blocks(self) -> bool {
        self.include_blocks != 0
    }

    fn wants_ports(self) -> bool {
        self.include_blocks != 0 && self.include_ports != 0
    }

    fn wants_connections(self) -> bool {
        self.include_connections != 0
    }
}

impl Default for DesVisualBlockFfiConfig {
    fn default() -> Self {
        Self::all_views()
    }
}

/// C ABI rectangle/layout value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DesVisualBlockFfiRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// C ABI visual port view.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DesVisualBlockFfiPort {
    pub id: *const c_char,
    pub direction: *const c_char,
    pub kind: *const c_char,
    pub label: *const c_char,
    /// Nullable.
    pub data_type: *const c_char,
    /// Boolean-as-byte.
    pub required: u8,
    /// Boolean-as-byte. When false, ignore `capacity`.
    pub has_capacity: u8,
    pub capacity: i64,
}

/// C ABI visual block view.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DesVisualBlockFfiBlock {
    pub id: *const c_char,
    pub kind: *const c_char,
    pub role: *const c_char,
    pub label: *const c_char,
    pub layout: DesVisualBlockFfiRect,
    pub fill: *const c_char,
    pub stroke: *const c_char,
    pub text: *const c_char,
    /// Nullable when `input_port_count == 0` or ports were not requested.
    pub input_ports: *const DesVisualBlockFfiPort,
    pub input_port_count: usize,
    /// Nullable when `output_port_count == 0` or ports were not requested.
    pub output_ports: *const DesVisualBlockFfiPort,
    pub output_port_count: usize,
}

/// C ABI visual connection view.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DesVisualBlockFfiConnection {
    pub id: *const c_char,
    pub kind: *const c_char,
    pub from_block_id: *const c_char,
    pub from_port_id: *const c_char,
    pub to_block_id: *const c_char,
    pub to_port_id: *const c_char,
}

/// Opaque owner for all FFI pointers.
///
/// Do not construct or inspect this type from C/C++/C#. Allocate it through one
/// of the constructor functions and release it with [`des_visual_block_ir_free`].
#[repr(C)]
pub struct DesVisualBlockIrHandle {
    json: Option<CString>,
    strings: Vec<CString>,
    ports: Vec<DesVisualBlockFfiPort>,
    blocks: Vec<DesVisualBlockFfiBlock>,
    connections: Vec<DesVisualBlockFfiConnection>,
}

impl DesVisualBlockIrHandle {
    /// Build an owned FFI handle from visual block specs.
    pub fn from_specs(specs: &[VisualBlockSpec], config: DesVisualBlockFfiConfig) -> Box<Self> {
        Box::new(build_handle(specs, config))
    }
}

/// Rust-side convenience for embedders that already have resolved specs.
pub fn visual_block_ir_ffi_handle(
    specs: &[VisualBlockSpec],
    config: DesVisualBlockFfiConfig,
) -> Box<DesVisualBlockIrHandle> {
    DesVisualBlockIrHandle::from_specs(specs, config)
}

/// Default FFI config: expose both JSON and pointer slices.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_default_config() -> DesVisualBlockFfiConfig {
    DesVisualBlockFfiConfig::all_views()
}

/// Pointer-first FFI config: expose block/port/connection slices, omit JSON.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_pointer_config() -> DesVisualBlockFfiConfig {
    DesVisualBlockFfiConfig::pointer_views()
}

/// JSON-only FFI config for legacy callers.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_json_config() -> DesVisualBlockFfiConfig {
    DesVisualBlockFfiConfig::json_only()
}

/// Schema id as a stable nul-terminated string.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_schema() -> *const c_char {
    VISUAL_BLOCK_IR_SCHEMA_C.as_ptr() as *const c_char
}

/// Current visual-block IR version.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_version() -> u32 {
    1
}

/// Construct a small sample graph so host bindings can smoke-test pointer
/// lifetimes without needing to manufacture Rust visual block specs.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_sample_new() -> *mut DesVisualBlockIrHandle {
    des_visual_block_ir_sample_new_with_config(DesVisualBlockFfiConfig::all_views())
}

/// Construct the sample graph with explicit view configuration.
#[no_mangle]
pub extern "C" fn des_visual_block_ir_sample_new_with_config(
    config: DesVisualBlockFfiConfig,
) -> *mut DesVisualBlockIrHandle {
    let specs = sample_visual_block_specs();
    Box::into_raw(DesVisualBlockIrHandle::from_specs(&specs, config))
}

/// Free a handle allocated by this module. Passing null is allowed.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_free(handle: *mut DesVisualBlockIrHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Nul-terminated JSON pointer. Returns null if JSON was not requested.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_json_ptr(
    handle: *const DesVisualBlockIrHandle,
) -> *const c_char {
    unsafe { handle_ref(handle) }
        .and_then(|h| h.json.as_ref())
        .map(|json| json.as_ptr())
        .unwrap_or(ptr::null())
}

/// JSON byte length excluding the trailing nul.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_json_len(
    handle: *const DesVisualBlockIrHandle,
) -> usize {
    unsafe { handle_ref(handle) }
        .and_then(|h| h.json.as_ref())
        .map(|json| json.as_bytes().len())
        .unwrap_or(0)
}

/// Number of blocks in the pointer view.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_block_count(
    handle: *const DesVisualBlockIrHandle,
) -> usize {
    unsafe { handle_ref(handle) }
        .map(|h| h.blocks.len())
        .unwrap_or(0)
}

/// Pointer to the first block view, or null when no block view exists.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_blocks_ptr(
    handle: *const DesVisualBlockIrHandle,
) -> *const DesVisualBlockFfiBlock {
    unsafe { handle_ref(handle) }
        .and_then(|h| ptr_or_null(&h.blocks))
        .unwrap_or(ptr::null())
}

/// Number of ports in the flat pointer view.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_port_count(
    handle: *const DesVisualBlockIrHandle,
) -> usize {
    unsafe { handle_ref(handle) }
        .map(|h| h.ports.len())
        .unwrap_or(0)
}

/// Pointer to the first flat port view, or null when no port view exists.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_ports_ptr(
    handle: *const DesVisualBlockIrHandle,
) -> *const DesVisualBlockFfiPort {
    unsafe { handle_ref(handle) }
        .and_then(|h| ptr_or_null(&h.ports))
        .unwrap_or(ptr::null())
}

/// Number of top-level visual connections in the pointer view.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_connection_count(
    handle: *const DesVisualBlockIrHandle,
) -> usize {
    unsafe { handle_ref(handle) }
        .map(|h| h.connections.len())
        .unwrap_or(0)
}

/// Pointer to the first connection view, or null when no connection view exists.
#[no_mangle]
pub unsafe extern "C" fn des_visual_block_ir_connections_ptr(
    handle: *const DesVisualBlockIrHandle,
) -> *const DesVisualBlockFfiConnection {
    unsafe { handle_ref(handle) }
        .and_then(|h| ptr_or_null(&h.connections))
        .unwrap_or(ptr::null())
}

unsafe fn handle_ref<'a>(
    handle: *const DesVisualBlockIrHandle,
) -> Option<&'a DesVisualBlockIrHandle> {
    if handle.is_null() {
        None
    } else {
        unsafe { handle.as_ref() }
    }
}

fn ptr_or_null<T>(items: &[T]) -> Option<*const T> {
    if items.is_empty() {
        None
    } else {
        Some(items.as_ptr())
    }
}

fn build_handle(
    specs: &[VisualBlockSpec],
    config: DesVisualBlockFfiConfig,
) -> DesVisualBlockIrHandle {
    let json = if config.wants_json() {
        Some(cstring_lossy(&visual_block_ir(specs).to_json_string()))
    } else {
        None
    };

    let mut strings = Vec::new();
    let mut ports = Vec::new();
    let mut blocks = Vec::new();
    let mut connections = Vec::new();

    if config.wants_ports() {
        let port_count = specs
            .iter()
            .map(|spec| spec.ports.inputs.len() + spec.ports.outputs.len())
            .sum();
        ports.reserve_exact(port_count);
    }
    if config.wants_blocks() {
        blocks.reserve_exact(specs.len());
    }
    if config.wants_connections() {
        let connection_count = specs.iter().map(|spec| spec.connections_out.len()).sum();
        connections.reserve_exact(connection_count);
    }

    let mut port_ranges = Vec::with_capacity(specs.len());
    if config.wants_ports() {
        for spec in specs {
            let input_start = ports.len();
            for port in &spec.ports.inputs {
                ports.push(port_to_ffi(port, &mut strings));
            }
            let input_count = ports.len() - input_start;

            let output_start = ports.len();
            for port in &spec.ports.outputs {
                ports.push(port_to_ffi(port, &mut strings));
            }
            let output_count = ports.len() - output_start;

            port_ranges.push((input_start, input_count, output_start, output_count));
        }
    } else {
        port_ranges.resize(specs.len(), (0, 0, 0, 0));
    }

    if config.wants_blocks() {
        let port_base = ports.as_ptr();
        for (spec, (input_start, input_count, output_start, output_count)) in
            specs.iter().zip(port_ranges.iter().copied())
        {
            blocks.push(block_to_ffi(
                spec,
                port_base,
                input_start,
                input_count,
                output_start,
                output_count,
                &mut strings,
            ));
        }
    }

    if config.wants_connections() {
        for connection in specs.iter().flat_map(|spec| spec.connections_out.iter()) {
            connections.push(DesVisualBlockFfiConnection {
                id: intern(&mut strings, &connection.id),
                kind: intern(&mut strings, &connection.kind),
                from_block_id: intern(&mut strings, &connection.from.block_id),
                from_port_id: intern(&mut strings, &connection.from.port_id),
                to_block_id: intern(&mut strings, &connection.to.block_id),
                to_port_id: intern(&mut strings, &connection.to.port_id),
            });
        }
    }

    DesVisualBlockIrHandle {
        json,
        strings,
        ports,
        blocks,
        connections,
    }
}

fn block_to_ffi(
    spec: &VisualBlockSpec,
    port_base: *const DesVisualBlockFfiPort,
    input_start: usize,
    input_count: usize,
    output_start: usize,
    output_count: usize,
    strings: &mut Vec<CString>,
) -> DesVisualBlockFfiBlock {
    DesVisualBlockFfiBlock {
        id: intern(strings, &spec.id),
        kind: intern(strings, &spec.kind),
        role: intern(strings, spec.role.as_str()),
        label: intern(strings, &spec.label),
        layout: layout_to_ffi(spec.layout),
        fill: intern(strings, &spec.style.fill),
        stroke: intern(strings, &spec.style.stroke),
        text: intern(strings, &spec.style.text),
        input_ports: offset_ptr(port_base, input_start, input_count),
        input_port_count: input_count,
        output_ports: offset_ptr(port_base, output_start, output_count),
        output_port_count: output_count,
    }
}

fn port_to_ffi(port: &VisualBlockPort, strings: &mut Vec<CString>) -> DesVisualBlockFfiPort {
    DesVisualBlockFfiPort {
        id: intern(strings, &port.id),
        direction: intern(strings, port.direction.as_str()),
        kind: intern(strings, &port.kind),
        label: intern(strings, &port.label),
        data_type: optional_intern(strings, port.data_type.as_deref()),
        required: u8::from(port.required),
        has_capacity: u8::from(port.capacity.is_some()),
        capacity: port.capacity.unwrap_or(0),
    }
}

fn layout_to_ffi(layout: ResolvedLayout) -> DesVisualBlockFfiRect {
    DesVisualBlockFfiRect {
        x: layout.x,
        y: layout.y,
        w: layout.w,
        h: layout.h,
    }
}

fn offset_ptr<T>(base: *const T, start: usize, count: usize) -> *const T {
    if count == 0 || base.is_null() {
        ptr::null()
    } else {
        unsafe { base.add(start) }
    }
}

fn optional_intern(strings: &mut Vec<CString>, value: Option<&str>) -> *const c_char {
    value
        .map(|value| intern(strings, value))
        .unwrap_or(ptr::null())
}

fn intern(strings: &mut Vec<CString>, value: &str) -> *const c_char {
    strings.push(cstring_lossy(value));
    strings
        .last()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null())
}

fn cstring_lossy(value: &str) -> CString {
    match CString::new(value) {
        Ok(value) => value,
        Err(_) => CString::new(value.replace('\0', " ")).expect("nul bytes replaced"),
    }
}

fn sample_visual_block_specs() -> Vec<VisualBlockSpec> {
    let mut source = VisualBlock::source(
        "ffi-source",
        vec![VisualPortInput::Opts(VisualPortOptions {
            id: "out".to_string(),
            kind: Some("token".to_string()),
            label: Some("tokens".to_string()),
            data_type: Some("Token".to_string()),
            ..Default::default()
        })],
        VisualBlockOptions {
            kind: Some("ffi-source".to_string()),
            label: Some("FFI Source".to_string()),
            style: Some(VisualBlockStyle {
                fill: Some("#ecfeff".to_string()),
                stroke: Some("#0891b2".to_string()),
                text: Some("#164e63".to_string()),
            }),
            ..Default::default()
        },
    );
    let mut sink = VisualBlock::sink(
        "ffi-sink",
        vec![VisualPortInput::Opts(VisualPortOptions {
            id: "in".to_string(),
            kind: Some("token".to_string()),
            label: Some("tokens".to_string()),
            data_type: Some("Token".to_string()),
            ..Default::default()
        })],
        VisualBlockOptions {
            kind: Some("ffi-sink".to_string()),
            label: Some("FFI Sink".to_string()),
            style: Some(VisualBlockStyle {
                fill: Some("#f0fdf4".to_string()),
                stroke: Some("#16a34a".to_string()),
                text: Some("#14532d".to_string()),
            }),
            ..Default::default()
        },
    );
    source.connect_to(
        &mut sink,
        VisualBlockConnectionOptions {
            kind: Some("token".to_string()),
            ..Default::default()
        },
    );

    visual_block_specs(&[&source, &sink])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::visual_block::VISUAL_BLOCK_IR_SCHEMA;
    use std::ffi::CStr;

    unsafe fn str_from_ptr(ptr: *const c_char) -> String {
        assert!(!ptr.is_null());
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn ffi_handle_exposes_json_and_pointer_views() {
        let handle = des_visual_block_ir_sample_new();
        assert!(!handle.is_null());

        unsafe {
            let schema = str_from_ptr(des_visual_block_ir_schema());
            assert_eq!(schema, VISUAL_BLOCK_IR_SCHEMA);
            assert_eq!(des_visual_block_ir_version(), 1);

            let json_ptr = des_visual_block_ir_json_ptr(handle);
            assert!(!json_ptr.is_null());
            assert!(des_visual_block_ir_json_len(handle) > 0);
            let json = str_from_ptr(json_ptr);
            assert!(json.contains("\"$schema\":\"des/visual-block-ir/v1\""));
            assert!(json.contains("\"ffi-source\""));

            let block_count = des_visual_block_ir_block_count(handle);
            assert_eq!(block_count, 2);
            let blocks =
                std::slice::from_raw_parts(des_visual_block_ir_blocks_ptr(handle), block_count);
            assert_eq!(str_from_ptr(blocks[0].id), "ffi-source");
            assert_eq!(str_from_ptr(blocks[0].role), "source");
            assert_eq!(blocks[0].output_port_count, 1);
            let output_ports =
                std::slice::from_raw_parts(blocks[0].output_ports, blocks[0].output_port_count);
            assert_eq!(str_from_ptr(output_ports[0].id), "out");
            assert_eq!(str_from_ptr(output_ports[0].kind), "token");
            assert_eq!(str_from_ptr(output_ports[0].data_type), "Token");

            let connection_count = des_visual_block_ir_connection_count(handle);
            assert_eq!(connection_count, 1);
            let connections = std::slice::from_raw_parts(
                des_visual_block_ir_connections_ptr(handle),
                connection_count,
            );
            assert_eq!(str_from_ptr(connections[0].from_block_id), "ffi-source");
            assert_eq!(str_from_ptr(connections[0].to_block_id), "ffi-sink");

            des_visual_block_ir_free(handle);
        }
    }

    #[test]
    fn ffi_handle_can_be_pointer_only() {
        let handle =
            des_visual_block_ir_sample_new_with_config(DesVisualBlockFfiConfig::pointer_views());
        assert!(!handle.is_null());

        unsafe {
            assert!(des_visual_block_ir_json_ptr(handle).is_null());
            assert_eq!(des_visual_block_ir_json_len(handle), 0);
            assert_eq!(des_visual_block_ir_block_count(handle), 2);
            assert_eq!(des_visual_block_ir_port_count(handle), 2);
            assert_eq!(des_visual_block_ir_connection_count(handle), 1);
            des_visual_block_ir_free(handle);
        }
    }

    #[test]
    fn ffi_handle_can_be_json_only() {
        let handle =
            des_visual_block_ir_sample_new_with_config(DesVisualBlockFfiConfig::json_only());
        assert!(!handle.is_null());

        unsafe {
            assert!(!des_visual_block_ir_json_ptr(handle).is_null());
            assert_eq!(des_visual_block_ir_block_count(handle), 0);
            assert!(des_visual_block_ir_blocks_ptr(handle).is_null());
            assert_eq!(des_visual_block_ir_port_count(handle), 0);
            assert_eq!(des_visual_block_ir_connection_count(handle), 0);
            des_visual_block_ir_free(handle);
        }
    }
}
