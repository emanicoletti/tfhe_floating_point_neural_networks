use half::f16;

pub struct PlainDenseLayer {
    weights: Vec<Vec<f16>>, // [output_size][input_size]
    biases: Vec<f16>,       // [output_size]
}

impl PlainDenseLayer {
    /// Creates a new dense layer with random or zero weights (here: zeros for simplicity)
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = vec![vec![f16::from_f32(1.0); input_size]; output_size];
        let biases = vec![f16::from_f32(0.0); output_size];
        Self { weights, biases }
    }

    pub fn train(
        &mut self,
        learning_rate: f32,
        _batch_size: usize,
        epochs: usize,
        train_inputs: &[Vec<f32>],
        train_labels: &[Vec<f32>],
    ) {
        assert_eq!(train_inputs[0].len(), self.weights[0].len(), "Input size does not match layer input dimension");

        let inputs_f16: Vec<Vec<f16>> = train_inputs
            .iter()
            .map(|input_vec| input_vec.iter().map(|&x| f16::from_f32(x)).collect())
            .collect();

        let labels_f16: Vec<Vec<f16>> = train_labels
            .iter()
            .map(|label_vec| label_vec.iter().map(|&x| f16::from_f32(x)).collect())
            .collect();

        let learning_rate_f16 = f16::from_f32(learning_rate);

        for epoch in 0..epochs {
            let outputs = self.forward_batch(&inputs_f16);

            let loss = PlainDenseLayer::mse_loss(&outputs, &labels_f16);
            println!("Epoch {} - Loss: {:.4}", epoch + 1, loss.to_f32());

            let grad_outputs = PlainDenseLayer::mse_gradient(&outputs, &labels_f16);

            self.backward_batch(&inputs_f16, &grad_outputs, learning_rate_f16);
            self.print_learned_parameters();
        }
    }

    fn forward(&self, input: &[f16]) -> Vec<f16> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(weight_row, &bias)| {
                let sum = weight_row
                    .iter()
                    .zip(input.iter())
                    .fold(f16::from_f32(0.0), |acc, (&w, &x)| acc + w * x);
                sum + bias
            })
            .collect()
    }

    fn forward_batch(&self, inputs: &[Vec<f16>]) -> Vec<Vec<f16>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    fn backward_batch(
        &mut self,
        inputs: &[Vec<f16>],
        grad_outputs: &[Vec<f16>],
        learning_rate: f16,
    ) {
        let batch_size = f16::from_f32(inputs.len() as f32);

        let output_dim = self.weights.len();
        let input_dim = self.weights[0].len();

        let mut grad_weights = vec![vec![f16::from_f32(0.0); input_dim]; output_dim];
        let mut grad_biases = vec![f16::from_f32(0.0); output_dim];

        for (input, grad_output) in inputs.iter().zip(grad_outputs.iter()) {
            for (neuron_idx, &grad_out_j) in grad_output.iter().enumerate() {
                grad_biases[neuron_idx] += grad_out_j;

                for (k, &x_i) in input.iter().enumerate() {
                    grad_weights[neuron_idx][k] += grad_out_j * x_i;
                }
            }
        }

        for i in 0..output_dim {
            for j in 0..input_dim {
                let avg_grad = grad_weights[i][j] / batch_size;
                self.weights[i][j] -= learning_rate * avg_grad;
            }
            let avg_bias_grad = grad_biases[i] / batch_size;
            self.biases[i] -= learning_rate * avg_bias_grad;
        }
    }

    pub fn print_learned_parameters(&self) {
        println!("\nLearned Weights:");
        for (i, row) in self.weights.iter().enumerate() {
            let decoded: Vec<f32> = row.iter().map(|w| w.to_f32()).collect();
            println!("Neuron {}: {:?}", i, decoded);
        }

        println!("\nLearned Biases:");
        let decoded_biases: Vec<f32> = self.biases.iter().map(|b| b.to_f32()).collect();
        println!("{:?}", decoded_biases);
    }

    pub fn mse_loss(predicted: &[Vec<f16>], target: &[Vec<f16>]) -> f16 {
        let batch_size = predicted.len();
        let output_dim = predicted[0].len();
        let mut sum = f16::from_f32(0.0);
    
        for (p, t) in predicted.iter().zip(target.iter()) {
            for i in 0..output_dim {
                let diff = p[i] - t[i];
                sum += diff * diff;
            }
        }
    
        let denom = f16::from_f32((batch_size * output_dim) as f32);
        sum / denom
    }

    pub fn mse_gradient(predicted: &[Vec<f16>], target: &[Vec<f16>]) -> Vec<Vec<f16>> {
        let batch_size = predicted.len();
        let output_dim = predicted[0].len();
        let denom = f16::from_f32((batch_size * output_dim) as f32);
        let two = f16::from_f32(2.0);
    
        let mut grads = vec![vec![f16::from_f32(0.0); output_dim]; batch_size];
    
        for (i, (p, t)) in predicted.iter().zip(target.iter()).enumerate() {
            for j in 0..output_dim {
                let diff = p[j] - t[j];
                grads[i][j] = (two * diff) / denom;
            }
        }
    
        grads
    }
}