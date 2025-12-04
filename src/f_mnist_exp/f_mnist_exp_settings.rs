use crate::plain_nn_builder::plain_nn::{PlainNeuralNetwork, PlainNeuralNetworkU32};
use crate::plain_nn_builder::plain_utils::load_from_file::*;

#[allow(dead_code)]
pub fn fmnist_exp1_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool) -> Result<(), Box<dyn std::error::Error>> {
    
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    // Load data from folder experiment_1_2 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(5)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(5));

    if train_plain_network {

        // Define the architecture
        plain_model.add_dense(784, 200);
        plain_model.add_tanh_activation(200);
        plain_model.add_dense(200, 10);
        plain_model.add_tanh_activation(10);

        plain_model.train_and_validate(
            20,
            50,
            0.5,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![50000, 1, 28, 28],
            vec![50000, 1, 1, 10],
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

#[allow(dead_code)]
pub fn fmnist_exp2_fp32(train_plain_network: bool, train_encrypted_network: bool, test_plain_network: bool) -> Result<(), Box<dyn std::error::Error>> {
    
    if !train_plain_network && test_plain_network {
        panic!("Cannot test plain network without training it first");
    }
    if !train_plain_network && !train_encrypted_network {
        panic!("At least one of train_plain_network or train_encrypted_network must be true");
    }

    // Load data from folder experiment_1_2 
    let (train_inputs_arr, train_labels_arr, _, _, test_inputs_arr, test_labels_arr) = load_data(5)?;

    // Format the input accordingly
    let train_inputs = array2_to_vecvec(&train_inputs_arr);
    let train_labels = array2_to_vecvec(&train_labels_arr);
    let test_inputs = array2_to_vecvec(&test_inputs_arr);
    let test_labels = array2_to_vecvec(&test_labels_arr);

    // Declare the plain model structure
    let mut plain_model = PlainNeuralNetworkU32::create(Some(6));

    if train_plain_network {

        // Define the architecture
        plain_model.add_conv(1, 6, 5, 5, 1, 2);
        plain_model.add_relu_activation(4704);
        plain_model.add_avg_pooling(vec![1, 6, 28, 28], 2, 2);
        plain_model.add_conv(6, 16, 5, 5, 1, 0);
        plain_model.add_relu_activation(1600);
        plain_model.add_avg_pooling(vec![1, 16, 10, 10], 2, 2);
        plain_model.add_dense(16*5*5, 120);
        plain_model.add_relu_activation(120);
        plain_model.add_dense(120, 84);
        plain_model.add_relu_activation(84);
        plain_model.add_dense(84, 10);

        plain_model.train_and_validate(
            20,
            50,
            2.0,
            &train_inputs,
            &train_labels,
            &test_inputs,
            &test_labels,
            vec![50000, 1, 28, 28],
            vec![50000, 1, 1, 10],
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