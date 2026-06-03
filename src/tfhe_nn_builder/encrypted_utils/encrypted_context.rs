use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use tfhe::ClientKey;

pub struct EncryptedContext<K: ServerKeyTrait, T> {
    pub encrypted_zero: T,
    pub encrypted_mask: T,
    pub client_key: ClientKey,
    pub server_key: K,
    pub ranges: Vec<(T, T, T, T, T)>, // piecewise segments: (min, max, a, b, derivative)
}
