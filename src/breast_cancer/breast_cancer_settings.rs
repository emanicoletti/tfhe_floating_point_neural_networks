use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn breast_cancer_fp32() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) =
        load_data(9)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Load weights and biases from file if needed
    let mut plain_model = PlainNeuralNetworkU32::create(Some(10));

    // Define the architecture
    plain_model.add_dense(30, 16);
    plain_model.add_relu_activation(16);
    plain_model.add_dense(16, 8);
    plain_model.add_relu_activation(8);
    plain_model.add_dense(8, 2);

    plain_model.train_and_validate(
        100,
        16,
        0.005,
        0.0001,
        0.9,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![455, 1, 1, 30],
        vec![455, 1, 1, 2],
        vec![114, 1, 1, 30],
        vec![114, 1, 1, 2],
    );

    Ok(())
}
