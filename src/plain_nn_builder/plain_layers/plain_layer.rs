pub trait PlainLayer: Send + Sync
{
    fn forward(&mut self, input: &Vec<u32>) -> Vec<u32>;

    fn backward(
        &mut self,
        input: &Vec<u32>,
        grad_output: &Vec<u32>,
    ) -> EncryptedTensor<T>;

    fn update_parameters(
        &mut self,
        learning_rate: f32,
    );

    fn get_weights(&self) -> Ve;
    
    fn get_biases(&self) -> EncryptedTensor<T>;

    fn get_grad_weights(&self) -> EncryptedTensor<T>;

    fn get_grad_biases(&self) -> EncryptedTensor<T>;

    fn get_id(&self) -> String;
}