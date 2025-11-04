use arrow_schema::DataType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DSLError {
    #[error("Invalid input index: {0}")]
    InvalidInputIndex(usize),

    #[error("Invalid kernel output length: {0}")]
    InvalidKernelOutputLength(usize),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Boolean expected: {0}")]
    BooleanExpected(String),

    #[error("Unused input: {0}")]
    UnusedInput(String),

    #[error("Unsupported dictionary value type: {0}")]
    UnsupportedDictionaryValueType(DataType),

    #[error("Unsupported list value type: {0}")]
    UnsupportedListValueType(DataType),

    #[error("No iteration")]
    NoIteration,

    #[error("Not vectorizable: {0}")]
    NotVectorizable(&'static str),
}
