use std::ops::Deref;

use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

#[derive(Debug)]
pub(crate) struct TokenBuffer2D<T>(pub(crate) Option<DMatrix<T>>);

#[derive(Debug)]
pub(crate) struct TokenBuffer4D<T>(pub(crate) Option<Vec<DMatrix<Vec<T>>>>);

impl<T> TokenBuffer2D<T> {
    pub(crate) fn new() -> Self {
        Self(None)
    }
}

impl<T> From<DMatrix<T>> for TokenBuffer2D<T> {
    fn from(matrix: DMatrix<T>) -> Self {
        Self(Some(matrix))
    }
}

impl<T: ToTokens> ToTokens for TokenBuffer2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let mut rows: Vec<TokenStream2> = Vec::new();
        for row in self.row_iter() {
            let iter = row.iter();
            rows.push(quote!(#(#iter),*));
        }
        let output = quote!(nalgebra::matrix![#(#rows);*]);
        output.to_tokens(tokens);
    }
}

impl<T> Deref for TokenBuffer2D<T> {
    type Target = DMatrix<T>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl<T> TokenBuffer4D<T> {
    pub(crate) fn new() -> Self {
        Self(None)
    }
}

impl<T> From<Vec<DMatrix<Vec<T>>>> for TokenBuffer4D<T> {
    fn from(data: Vec<DMatrix<Vec<T>>>) -> Self {
        Self(Some(data))
    }
}

impl<T: ToTokens> ToTokens for TokenBuffer4D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let mut batches: Vec<TokenStream2> = Vec::new();
        for batch in self.iter() {
            let mut rows: Vec<TokenStream2> = Vec::new();
            for row in batch.row_iter() {
                let mut elements: Vec<TokenStream2> = Vec::new();
                for element in row.iter() {
                    let iter = element.iter();
                    elements.push(quote!([#(#iter),*]));
                }
                rows.push(quote!(#(#elements),*));
            }
            batches.push(quote!(nalgebra::matrix![#(#rows);*]));
        }
        let output = quote!([#(#batches),*]);
        output.to_tokens(tokens);
    }
}

impl<T> Deref for TokenBuffer4D<T> {
    type Target = Vec<DMatrix<Vec<T>>>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    #[test]
    fn buffer_2d_new() {
        assert_eq!(TokenBuffer2D::<i8>::new().0, None);
    }

    #[test]
    fn buffer_2d_from_matrix() {
        let matrix = dmatrix![1, 2, 3];
        assert_eq!(TokenBuffer2D::from(matrix.clone()).0, Some(matrix));
    }

    #[test]
    fn buffer_2d_to_tokens() {
        let buffer = TokenBuffer2D::from(dmatrix![
            1i8, 2i8, 3i8;
            4i8, 5i8, 6i8
        ]);
        assert_eq!(
            buffer.to_token_stream().to_string(),
            quote! {
                nalgebra::matrix![
                    1i8, 2i8, 3i8;
                    4i8, 5i8, 6i8
                ]
            }
            .to_string()
        );
    }

    #[test]
    fn buffer_4d_new() {
        assert_eq!(TokenBuffer4D::<i8>::new().0, None);
    }

    #[test]
    fn buffer_4d_from_data() {
        let data = vec![dmatrix![vec![1], vec![2], vec![3]]];
        assert_eq!(TokenBuffer4D::from(data.clone()).0, Some(data));
    }

    #[test]
    fn buffer_4d_to_tokens() {
        let buffer = TokenBuffer4D::from(vec![
            dmatrix![
                vec![1i8, 2i8], vec![3i8, 4i8],  vec![5i8, 6i8];
                vec![7i8, 8i8], vec![9i8, 10i8], vec![11i8, 12i8]
            ],
            dmatrix![
                vec![13i8, 14i8], vec![15i8, 16i8], vec![17i8, 18i8];
                vec![19i8, 20i8], vec![21i8, 22i8], vec![23i8, 24i8]
            ],
        ]);
        assert_eq!(
            buffer.to_token_stream().to_string(),
            quote! {
                [
                    nalgebra::matrix![
                        [1i8, 2i8], [3i8, 4i8],  [5i8,  6i8];
                        [7i8, 8i8], [9i8, 10i8], [11i8, 12i8]
                    ],
                    nalgebra::matrix![
                        [13i8, 14i8], [15i8, 16i8], [17i8, 18i8];
                        [19i8, 20i8], [21i8, 22i8], [23i8, 24i8]
                    ]
                ]
            }
            .to_string()
        );
    }
}
