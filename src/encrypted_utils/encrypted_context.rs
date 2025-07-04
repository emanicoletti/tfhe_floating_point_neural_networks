use crate::encrypted_utils::server_key_trait::ServerKeyTrait;

pub struct EncryptedContext<K: ServerKeyTrait, T> {
    pub encrypted_zero: T,
    pub encrypted_mask: T,
    pub server_key: K,
}