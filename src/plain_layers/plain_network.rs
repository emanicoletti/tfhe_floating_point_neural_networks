use half::f16;

pub struct PlainDenseLayer {
    pub weights: Vec<Vec<f16>>, // [output_size][input_size]
    pub biases: Vec<f16>,       // [output_size]
}

impl PlainDenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = vec![vec![f16::from_f32(1.0); input_size]; output_size];
        let biases = vec![f16::from_f32(0.0); output_size];
        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f16]) -> Vec<f16> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, &bias)| {
                w_row
                    .iter()
                    .zip(input.iter())
                    .fold(f16::from_f32(0.0), |acc, (&w, &x)| acc + w * x)
                    + bias
            })
            .collect()
    }

    pub fn forward_batch(&self, inputs: &[Vec<f16>]) -> Vec<Vec<f16>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    pub fn backward_batch(
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
            for (j, &grad_out_j) in grad_output.iter().enumerate() {
                grad_biases[j] += grad_out_j;
                for (k, &x_k) in input.iter().enumerate() {
                    grad_weights[j][k] += grad_out_j * x_k;
                }
            }
        }

        for j in 0..output_dim {
            for k in 0..input_dim {
                self.weights[j][k] -= learning_rate * (grad_weights[j][k] / batch_size);
            }
            self.biases[j] -= learning_rate * (grad_biases[j] / batch_size);
        }
    }

    pub fn print_learned_parameters(&self) {
        println!("Weights:");
        for (i, row) in self.weights.iter().enumerate() {
            let decoded: Vec<f32> = row.iter().map(|w| w.to_f32()).collect();
            println!(" Neuron {}: {:?}", i, decoded);
        }

        println!("Biases:");
        let decoded: Vec<f32> = self.biases.iter().map(|b| b.to_f32()).collect();
        println!(" {:?}", decoded);
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
        sum / f16::from_f32((batch_size * output_dim) as f32)
    }

    pub fn mse_gradient(predicted: &[Vec<f16>], target: &[Vec<f16>]) -> Vec<Vec<f16>> {
        let batch_size = predicted.len();
        let output_dim = predicted[0].len();
        let denom = f16::from_f32((batch_size * output_dim) as f32);
        let two = f16::from_f32(2.0);

        predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| {
                (0..output_dim)
                    .map(|j| (two * (p[j] - t[j])) / denom)
                    .collect()
            })
            .collect()
    }
}

pub struct PlainNetwork {
    pub layers: Vec<PlainDenseLayer>,
}

impl PlainNetwork {
    pub fn new(layer_dims: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_dims.len() - 1 {
            layers.push(PlainDenseLayer::new(layer_dims[i], layer_dims[i + 1]));
        }
        Self { layers }
    }

    pub fn train(
        &mut self,
        learning_rate: f32,
        batch_size: usize,
        epochs: usize,
        train_inputs: &[Vec<f32>],
        train_labels: &[Vec<f32>],
    ) {
        let total_samples = train_inputs.len();
        let lr = f16::from_f32(learning_rate);

        for epoch in 0..epochs {
            println!("Epoch {}", epoch + 1);

            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);

                let input_batch: Vec<Vec<f16>> = train_inputs[batch_start..batch_end]
                    .iter()
                    .map(|v| v.iter().map(|&x| f16::from_f32(x)).collect())
                    .collect();

                let label_batch: Vec<Vec<f16>> = train_labels[batch_start..batch_end]
                    .iter()
                    .map(|v| v.iter().map(|&x| f16::from_f32(x)).collect())
                    .collect();

                // Forward pass
                let mut activations: Vec<Vec<Vec<f16>>> = vec![input_batch.clone()];
                for layer in &self.layers {
                    let layer_out: Vec<Vec<f16>> = activations.last().unwrap()
                        .iter()
                        .map(|a| layer.forward(a))
                        .collect();
                    activations.push(layer_out);
                }

                let predictions = activations.last().unwrap();
                let loss = PlainDenseLayer::mse_loss(predictions, &label_batch);
                println!("  Batch Loss: {:.4}", loss.to_f32());

                let mut grad = PlainDenseLayer::mse_gradient(predictions, &label_batch);

                for l in (0..self.layers.len()).rev() {
                    let input_to_layer = &activations[l];
                    self.layers[l].backward_batch(input_to_layer, &grad, lr);

                    if l > 0 {
                        let w_t = transpose(&self.layers[l].weights);
                        grad = input_to_layer
                            .iter()
                            .map(|_| {
                                w_t.iter()
                                    .map(|w_row| dot(w_row, &grad[0]))
                                    .collect()
                            })
                            .collect();
                    }
                }
            }
        }

        self.print_parameters();
    }

    pub fn print_parameters(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("\nLayer {}:", i);
            layer.print_learned_parameters();
        }
    }
}

fn dot(a: &[f16], b: &[f16]) -> f16 {
    a.iter().zip(b.iter()).fold(f16::from_f32(0.0), |acc, (&x, &y)| acc + x * y)
}

fn transpose(matrix: &[Vec<f16>]) -> Vec<Vec<f16>> {
    if matrix.is_empty() { return vec![]; }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![f16::from_f32(0.0); rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}
