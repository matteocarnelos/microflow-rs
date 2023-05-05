use crate::tflite_flatbuffers::tflite::ActivationFunctionType;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct TokenFusedActivation(pub(crate) ActivationFunctionType);

impl ToTokens for TokenFusedActivation {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let output = match self.0 {
            ActivationFunctionType::NONE => quote!(microflow::activation::FusedActivation::NONE),
            ActivationFunctionType::RELU => quote!(microflow::activation::FusedActivation::RELU),
            ActivationFunctionType::RELU6 => quote!(microflow::activation::FusedActivation::RELU6),
            _ => unimplemented!(),
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_to_tokens() {
        let activation = TokenFusedActivation(ActivationFunctionType::RELU);
        assert_eq!(
            activation.to_token_stream().to_string(),
            quote!(microflow::activation::FusedActivation::RELU).to_string()
        );
    }
}
