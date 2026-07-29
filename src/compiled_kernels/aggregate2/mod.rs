use arrow_array::{Array, ArrayRef, UInt64Array};
use arrow_schema::DataType;

use crate::{compiled_kernels::aggregate2::minmax::MinMaxAggregator, ArrowKernelError};

mod count;
mod minmax;
mod most_recent;
mod sum;

pub use count::CountAggregator;
pub use most_recent::MostRecentAggregator;
pub use sum::SumAggregator;
pub type MinAggregator = MinMaxAggregator<true>;
pub type MaxAggregator = MinMaxAggregator<false>;

/// Stateful aggregation interface for grouped and ungrouped kernels.
///
/// Implementations expect callers to reserve enough backing storage before
/// ingestion. For grouped aggregation, capacity must be at least
/// `max_ticket + 1`; for ungrouped aggregation, reserve one slot when the
/// input is non-empty.
pub trait Aggregator {
    /// Creates a new aggregator for the given input types.
    ///
    /// Implementations validate the input arity and return an error when the
    /// provided types are incompatible with the aggregation.
    fn create(tys: &[&DataType]) -> Result<Box<Self>, ArrowKernelError>;

    /// Ensures the aggregator has storage for at least `capacity` groups.
    fn ensure_capacity(&mut self, capacity: usize);

    /// Ingests one batch of grouped input using `tickets` to select the target
    /// group for each row.
    ///
    /// ```
    /// use arrow_array::{cast::AsArray, types::UInt64Type, Int32Array, UInt64Array};
    /// use arrow_compile_compute::aggregate::{self, Aggregator};
    ///
    /// let arr = Int32Array::from(vec![10, 20, 30, 40]);
    /// let tickets = UInt64Array::from(vec![0, 1, 0, 1]);
    ///
    /// let mut agg = aggregate::count().unwrap();
    /// agg.ingest(&[&arr], &tickets).unwrap();
    ///
    /// let result = agg.finish().unwrap();
    /// let result = result.as_primitive::<UInt64Type>();
    /// assert_eq!(result.values(), &[2, 2]);
    /// ```
    fn ingest(
        &mut self,
        data: &[&dyn Array],
        tickets: &UInt64Array,
    ) -> Result<(), ArrowKernelError>;

    /// Merges another aggregator of the same type into `self`.
    ///
    /// Implementations may grow one side to match the other before combining
    /// the partial results.
    fn merge(&mut self, other: Self) -> Result<(), ArrowKernelError>;

    fn ensure_capacity_for_tickets(&mut self, tickets: &UInt64Array) {
        if tickets.is_empty() {
            return;
        }

        let req_capacity = tickets.iter().map(|x| x.unwrap()).max().unwrap_or(0) + 1;
        self.ensure_capacity(req_capacity as usize);
    }

    /// Finalizes the aggregation and returns the output array.
    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError>;

    /// Ingests one ungrouped batch by routing every row to ticket `0`.
    ///
    /// ```
    /// use arrow_array::{cast::AsArray, types::Int32Type, Array, Int32Array};
    /// use arrow_compile_compute::aggregate::{self, Aggregator};
    ///
    /// let arr = Int32Array::from(vec![7, 2, 5]);
    /// let mut agg = aggregate::min(arr.data_type()).unwrap();
    /// agg.ingest_ungrouped(&[&arr]).unwrap();
    ///
    /// let result = agg.finish().unwrap();
    /// let result = result.as_primitive::<Int32Type>();
    /// assert_eq!(result.len(), 1);
    /// assert_eq!(result.value(0), 2);
    /// ```
    fn ingest_ungrouped(&mut self, data: &[&dyn Array]) -> Result<(), ArrowKernelError> {
        let tickets = UInt64Array::from(vec![0_u64; data[0].len()]);
        self.ingest(data, &tickets)
    }
}
