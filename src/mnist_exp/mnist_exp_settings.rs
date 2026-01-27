use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn mnist_exp_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool) -> Result<(), Box<dyn std::error::Error>> {
    
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    // Load data from folder experiment_1_2 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(4)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(4));

    if train_plain_network {

        // Define the architecture
        plain_model.add_conv(1, 6, 3, 3, 1, 0);
        plain_model.add_relu_activation(4056);
        plain_model.add_conv(6, 16, 3, 3, 1, 0);
        plain_model.add_relu_activation(9216);
        plain_model.add_dense(9216, 84);
        plain_model.add_relu_activation(84);
        plain_model.add_dense(84, 10);

        // Train and validate the model
        plain_model.train_and_validate(
            5,
            60,
            1.0,
            0.0,
            0.0,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![60000, 1, 28, 28],
            vec![60000, 1, 1, 10],
            vec![10000, 1, 28, 28],
            vec![10000, 1, 1, 10],
        );

    }

    if test_plain_network {

        // Validate the plain model
        let mut correct = 0;
        let total = test_labels.len();

        for (input, label) in test_inputs.iter().zip(test_labels.iter()) {
            let input_batch = vec![input.clone()]; 
            let input_shape = vec![1, 1, 28, 28];
            let label_shape = vec![1, 1, 1, 10];

            let prediction = plain_model.inference(&input_batch, input_shape, label_shape);

            let predicted_class = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let actual_class = label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_class == actual_class {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total as f32;
        println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    }

    Ok(())
}