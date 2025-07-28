pub struct DenseLayer {
    pub id: String,
    pub weights: Vec<Vec<u32>>, // shape: [input_dim][output_dim]
    pub biases: Vec<u32>,       // shape: [output_dim]
    pub grad_weights: Option<Vec<Vec<u32>>>,
    pub grad_biases: Option<Vec<u32>>,
}

impl DenseLayer {
    pub fn new(id: String, weights: Vec<Vec<u32>>, biases: Vec<u32>) -> Self {
        Self {
            id,
            weights,
            biases,
            grad_weights: None,
            grad_biases: None,
        }
    }

    pub fn forward(&self, input: &Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let batch_size = input.len();
        let output_dim = self.biases.len();

        let mut output = vec![vec![0u32; output_dim]; batch_size];

        for i in 0..batch_size {
            for j in 0..output_dim {
                // Start from bias (u32)
                let mut sum = self.biases[j];
                for k in 0..self.weights.len() {
                    // Multiplying u32, might overflow — consider using checked_mul if necessary
                    sum = sum.saturating_add(input[i][k].saturating_mul(self.weights[k][j]));
                }
                output[i][j] = sum;
            }
        }

        output
    }

    pub fn backward(&mut self, input: &Vec<Vec<u32>>, grad_output: &Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let batch_size = input.len();
        let input_dim = input[0].len();
        let output_dim = grad_output[0].len();

        // Compute grad_weights
        let mut grad_weights = vec![vec![0u32; output_dim]; input_dim];
        for i in 0..input_dim {
            for j in 0..output_dim {
                let mut acc = 0u32;
                for b in 0..batch_size {
                    acc = acc.saturating_add(input[b][i].saturating_mul(grad_output[b][j]));
                }
                grad_weights[i][j] = acc;
            }
        }

        // Compute grad_biases
        let mut grad_biases = vec![0u32; output_dim];
        for j in 0..output_dim {
            let mut acc = 0u32;
            for b in 0..batch_size {
                acc = acc.saturating_add(grad_output[b][j]);
            }
            grad_biases[j] = acc;
        }

        // Compute grad_input
        let mut grad_input = vec![vec![0u32; input_dim]; batch_size];
        for b in 0..batch_size {
            for i in 0..input_dim {
                let mut acc = 0u32;
                for j in 0..output_dim {
                    acc = acc.saturating_add(grad_output[b][j].saturating_mul(self.weights[i][j]));
                }
                grad_input[b][i] = acc;
            }
        }

        self.grad_weights = Some(grad_weights);
        self.grad_biases = Some(grad_biases);

        grad_input
    }

    pub fn update_parameters(&mut self, learning_rate: u32) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_biases) {
            for i in 0..self.weights.len() {
                for j in 0..self.weights[0].len() {
                    // saturating_sub to avoid panic on underflow
                    self.weights[i][j] = self.weights[i][j].saturating_sub(learning_rate.saturating_mul(grad_w[i][j]));
                }
            }

            for j in 0..self.biases.len() {
                self.biases[j] = self.biases[j].saturating_sub(learning_rate.saturating_mul(grad_b[j]));
            }
        }
    }

    pub fn get_weights(&self) -> Vec<Vec<u32>> {
        self.weights.clone()
    }

    pub fn get_biases(&self) -> Vec<u32> {
        self.biases.clone()
    }

    pub fn get_grad_weights(&self) -> Vec<Vec<u32>> {
        self.grad_weights.clone().unwrap()
    }

    pub fn get_grad_biases(&self) -> Vec<u32> {
        self.grad_biases.clone().unwrap()
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }
}