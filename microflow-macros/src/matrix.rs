use std::ops::Deref;

use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

#[derive(Debug)]
pub struct TokenMatrix<T> {
    pub(crate) matrix: DMatrix<T>,
}

impl<T> From<DMatrix<T>> for TokenMatrix<T> {
    fn from(matrix: DMatrix<T>) -> Self {
        Self { matrix }
    }
}

impl<T> ToTokens for TokenMatrix<T>
where
    T: ToTokens,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let mut matrix = TokenStream2::new();
        for row in self.row_iter() {
            let iter = row.iter();
            quote!(#(#iter),*;).to_tokens(&mut matrix);
        }

        let output = quote! { nalgebra::matrix![#matrix] };
        output.to_tokens(tokens);
    }
}

impl<T> Deref for TokenMatrix<T> {
    type Target = DMatrix<T>;
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}
