use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn sc_mnist_exp_fp32() -> Result<(), Box<dyn std::error::Error>> {
    
    // Load data from folder 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(6)?;

    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    let mut plain_model = PlainNeuralNetworkU32::create(Some(7));

        // Define the architecture
        plain_model.add_conv(3, 64, 3, 3, 1, 0);
        plain_model.add_batch_norm(64);
        plain_model.add_relu_activation(43264);
        plain_model.add_avg_pooling(vec![26, 26], 2, 2);
        plain_model.add_conv(64, 96, 3, 3, 1, 0);
        plain_model.add_batch_norm(96);
        plain_model.add_relu_activation(11616);
        plain_model.add_avg_pooling(vec![11, 11], 2, 2);
        plain_model.add_dense(2400, 128);
        plain_model.add_relu_activation(128);
        plain_model.add_dense(128, 7);
        plain_model.add_relu_activation(7);

        // Train and validate the model
        plain_model.train_and_validate(
            15,
            60,
            0.05,
            0.0,
            0.0,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![8010, 3, 28, 28],
            vec![8010, 1, 1, 7],
            vec![2005, 3, 28, 28],
            vec![2005, 1, 1, 7],
        );

    Ok(())
    
}