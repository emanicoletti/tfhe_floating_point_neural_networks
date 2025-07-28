pub struct PlainTensor<T: EncryptedElement> {
    pub data: Vec<T>,
    pub shape: Vec<usize>, // [batch, channels, height, width]
}
