use extendr_api::prelude::*;

/// add two integers supplied to rust
/// @export
#[extendr]
fn add_nums(x: i32, y: i32) -> i32 {
    return x + y;
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod tokenizeRs;
    fn add_nums;
}
