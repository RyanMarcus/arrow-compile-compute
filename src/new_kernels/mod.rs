mod cmp;

pub use cmp::ComparisonKernel;

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch,
}
