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
    ///
    /// This must be called before `ingest` when tickets may address groups that
    /// are not yet backed by internal buffers.
    fn ensure_capacity(&mut self, capacity: usize);

    /// Checks, in debug and test builds, that the caller reserved enough
    /// capacity for the provided ticket array.
    ///
    /// Release builds skip this check entirely.
    fn debug_assert_capacity_for_tickets(&self, tickets: &UInt64Array, capacity: usize) {
        #[cfg(any(debug_assertions, test))]
        if let Some(max_ticket) = tickets.values().iter().copied().max() {
            let required_capacity = usize::try_from(max_ticket + 1)
                .expect("ticket index exceeds addressable buffer size");
            debug_assert!(
                capacity >= required_capacity,
                "aggregator capacity {} is smaller than required ticket capacity {}",
                capacity,
                required_capacity
            );
        }
    }

    /// Ingests one batch of grouped input using `tickets` to select the target
    /// group for each row.
    ///
    /// Callers are responsible for reserving enough capacity before calling
    /// this method.
    ///
    /// ```
    /// use arrow_array::{cast::AsArray, types::UInt64Type, Int32Array, UInt64Array};
    /// use arrow_compile_compute::aggregate::{self, Aggregator};
    ///
    /// let arr = Int32Array::from(vec![10, 20, 30, 40]);
    /// let tickets = UInt64Array::from(vec![0, 1, 0, 1]);
    ///
    /// let mut agg = aggregate::count().unwrap();
    /// agg.ensure_capacity(2);
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

    /// Finalizes the aggregation and returns the output array.
    ///
    /// Consumes the aggregator because implementations may need to move or
    /// reinterpret internal buffers during finalization.
    fn finish(self: Box<Self>) -> Result<ArrayRef, ArrowKernelError>;

    /// Ingests one ungrouped batch by routing every row to ticket `0`.
    ///
    /// ```
    /// use arrow_array::{cast::AsArray, types::Int32Type, Array, Int32Array};
    /// use arrow_compile_compute::aggregate::{self, Aggregator};
    ///
    /// let arr = Int32Array::from(vec![7, 2, 5]);
    /// let mut agg = aggregate::min(arr.data_type()).unwrap();
    /// agg.ensure_capacity(1);
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
