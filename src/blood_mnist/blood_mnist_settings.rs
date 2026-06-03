use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn blood_vgg_fp32() -> Result<(), Box<dyn std::error::Error>> {
    // Load data from folder
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) =
        load_data(7)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Load weights and biases from file if needed
    let mut plain_model = PlainNeuralNetworkU32::create(Some(8));

    // Define the architecture
    // --- BLOCK 1 ---
    // Input: 28x28x3
    plain_model.add_conv(3, 64, 3, 3, 1, 0);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(43264);

    plain_model.add_conv(64, 64, 3, 3, 1, 0);
    plain_model.add_batch_norm(64);
    plain_model.add_relu_activation(36864);

    plain_model.add_max_pooling(vec![24, 24], 2, 2);

    // --- BLOCK 2 ---
    plain_model.add_conv(64, 96, 3, 3, 1, 0);
    plain_model.add_batch_norm(96);
    plain_model.add_relu_activation(9600);

    plain_model.add_conv(96, 96, 3, 3, 1, 0);
    plain_model.add_batch_norm(96);
    plain_model.add_relu_activation(6144);

    plain_model.add_max_pooling(vec![8, 8], 2, 2);

    // --- CLASSIFIER ---
    plain_model.add_dense(1536, 128);
    plain_model.add_relu_activation(128);
    plain_model.add_dense(128, 8);

    plain_model.train_and_validate(
        20,
        64,
        0.1,
        0.0,
        0.0,
        &train_inputs,
        &train_labels,
        &test_inputs,
        &test_labels,
        vec![13671, 3, 28, 28],
        vec![13671, 1, 1, 8],
        vec![3421, 3, 28, 28],
        vec![3421, 1, 1, 8],
    );

    Ok(())
}
