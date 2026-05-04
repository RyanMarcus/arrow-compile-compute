use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder, Int32Builder},
    Float32Array, Scalar,
};
use arrow_compile_compute::vec::nearest_neighbor;
use proptest::prelude::*;

fn f32_vectors(rows: &[Vec<f32>], dim: usize) -> arrow_array::FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for row in rows {
        assert_eq!(row.len(), dim);
        builder.values().append_slice(row);
        builder.append(true);
    }
    builder.finish()
}

fn i32_vectors(rows: &[Vec<i32>], dim: usize) -> arrow_array::FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(Int32Builder::new(), dim as i32);
    for row in rows {
        assert_eq!(row.len(), dim);
        builder.values().append_slice(row);
        builder.append(true);
    }
    builder.finish()
}

fn scalar_f32_vector(row: &[f32]) -> Scalar<arrow_array::FixedSizeListArray> {
    Scalar::new(f32_vectors(&[row.to_vec()], row.len()))
}

fn squared_norms(rows: &[Vec<f32>]) -> Float32Array {
    Float32Array::from(
        rows.iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f32>())
            .collect::<Vec<_>>(),
    )
}

fn reference_nearest(query: &[f32], values: &[Vec<f32>], norms: &[f32]) -> Option<u64> {
    let mut best_idx = None;
    let mut best_dist = f32::INFINITY;
    for (idx, (value, norm)) in values.iter().zip(norms.iter()).enumerate() {
        let dot = query
            .iter()
            .zip(value.iter())
            .map(|(q, v)| q * v)
            .sum::<f32>();
        let dist = norm - 2.0 * dot;
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx as u64);
        }
    }
    best_idx
}

#[test]
fn test_nearest_neighbor_known_values() {
    let values = vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![-2.0, 0.0]];
    let query = scalar_f32_vector(&[0.0, 1.5]);
    let values_arr = f32_vectors(&values, 2);
    let norms = squared_norms(&values);

    assert_eq!(
        nearest_neighbor(&query, &values_arr, &norms).unwrap(),
        Some(1)
    );
}

#[test]
fn test_nearest_neighbor_empty_values() {
    let query = scalar_f32_vector(&[1.0, 2.0]);
    let values = f32_vectors(&[], 2);
    let norms = Float32Array::from(Vec::<f32>::new());

    assert_eq!(nearest_neighbor(&query, &values, &norms).unwrap(), None);
}

#[test]
fn test_nearest_neighbor_first_tie() {
    let values = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let query = scalar_f32_vector(&[1.0, 0.0]);
    let values_arr = f32_vectors(&values, 2);
    let norms = squared_norms(&values);

    assert_eq!(
        nearest_neighbor(&query, &values_arr, &norms).unwrap(),
        Some(0)
    );
}

#[test]
fn test_nearest_neighbor_rejects_bad_inputs() {
    let query = scalar_f32_vector(&[1.0, 2.0]);
    let values = f32_vectors(&[vec![1.0, 2.0]], 2);
    let norms = Float32Array::from(vec![5.0]);

    let query_array = f32_vectors(&[vec![1.0, 2.0]], 2);
    assert!(nearest_neighbor(&query_array, &values, &norms).is_err());

    let wrong_dim_values = f32_vectors(&[vec![1.0, 2.0, 3.0]], 3);
    assert!(nearest_neighbor(&query, &wrong_dim_values, &norms).is_err());

    let wrong_norms = Float32Array::from(vec![1.0, 2.0]);
    assert!(nearest_neighbor(&query, &values, &wrong_norms).is_err());

    let int_values = i32_vectors(&[vec![1, 2]], 2);
    assert!(nearest_neighbor(&query, &int_values, &norms).is_err());

    let nullable_norms = Float32Array::from(vec![Some(5.0), None]);
    let two_values = f32_vectors(&[vec![1.0, 2.0], vec![2.0, 1.0]], 2);
    assert!(nearest_neighbor(&query, &two_values, &nullable_norms).is_err());

    let nullable_query = Scalar::new({
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(false);
        builder.finish()
    });
    assert!(nearest_neighbor(&nullable_query, &values, &norms).is_err());
}

#[test]
fn test_nearest_neighbor_rejects_child_nulls() {
    let query = Scalar::new({
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        builder.values().append_option(Some(1.0));
        builder.values().append_option(None);
        builder.append(true);
        builder.finish()
    });
    let values = f32_vectors(&[vec![1.0, 2.0]], 2);
    let norms = Float32Array::from(vec![5.0]);
    assert!(nearest_neighbor(&query, &values, &norms).is_err());

    let query = scalar_f32_vector(&[1.0, 2.0]);
    let values = {
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        builder.values().append_option(Some(1.0));
        builder.values().append_option(None);
        builder.append(true);
        builder.finish()
    };
    assert!(nearest_neighbor(&query, &values, &norms).is_err());
}

proptest! {
    #[test]
    fn test_nearest_neighbor_matches_reference(
        query_raw in prop::collection::vec(-100i16..100, 1..9),
        values_raw in prop::collection::vec(prop::collection::vec(-100i16..100, 1..9), 0..32),
    ) {
        let dim = query_raw.len();
        let values_raw = values_raw
            .into_iter()
            .map(|mut row| {
                row.resize(dim, 0);
                row.truncate(dim);
                row
            })
            .collect::<Vec<_>>();
        let query = query_raw.iter().map(|v| *v as f32 / 10.0).collect::<Vec<_>>();
        let values = values_raw
            .iter()
            .map(|row| row.iter().map(|v| *v as f32 / 10.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let norms_vec = squared_norms(&values);
        let expected = reference_nearest(
            &query,
            &values,
            norms_vec.values(),
        );

        let query = scalar_f32_vector(&query);
        let values_arr = f32_vectors(&values, dim);
        prop_assert_eq!(nearest_neighbor(&query, &values_arr, &norms_vec).unwrap(), expected);
    }
}
