mod cmp;

use arrow_schema::DataType;
pub use cmp::ComparisonKernel;

#[derive(Debug)]
pub enum ArrowKernelError {
    SizeMismatch,
    ArgumentMismatch,
    UnsupportedArguments(String),
    UnsupportedScalar(DataType),
}
