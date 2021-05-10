use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

#[proc_macro_attribute]
pub fn pyclass(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = syn::parse_macro_input!(input as syn::ItemStruct);

    let vis = &item.vis;

    let inner_ident = &item.ident;
    let cls_name = inner_ident.to_string();
    let py_ident = Ident::new(&format!("Py{}", inner_ident), Span::call_site());
    let dyn_ident = Ident::new(&format!("Dyn{}", inner_ident), Span::call_site());

    quote!(
        #[pyclass(name = #cls_name)]
        #vis struct #py_ident {
            wrapped: Box<dyn #dyn_ident>,
        }

        #item
    )
    .into()
}

fn get_inner_ident(self_ty: &syn::Type) -> syn::Result<&syn::Ident> {
    match self_ty {
        syn::Type::Path(type_path) => Ok(type_path
            .path
            .segments
            .iter()
            .last()
            .map(|segment| &segment.ident)
            .unwrap()),
        _ => Err(syn::Error::new_spanned(
            self_ty,
            "invalid `Self` type: should be a path",
        )),
    }
}

fn generate_pymethods(item: syn::ItemImpl) -> syn::Result<TokenStream> {
    let inner_ident = get_inner_ident(&item.self_ty)?;

    let self_ty = &item.self_ty;

    let (impl_generics, _, where_clause) = &item.generics.split_for_impl();

    let py_ident = Ident::new(&format!("Py{}", inner_ident), Span::call_site());
    let dyn_ident = Ident::new(&format!("Dyn{}", inner_ident), Span::call_site());

    let mut methods = Vec::new();
    for item in &item.items {
        match item {
            syn::ImplItem::Method(method) => methods.push(method),
            _ => return Err(syn::Error::new_spanned(item, "invalid non-method item")),
        }
    }

    let trait_methods = methods.iter().map(|method| &method.sig).collect::<Vec<_>>();

    let py_methods = methods.iter().map(|method| {
        let vis = &method.vis;
        let sig = &method.sig;
        let ident = &method.sig.ident;

        let mut self_receiver = None;

        let arguments = sig
            .inputs
            .iter()
            .filter_map(|arg| match arg {
                syn::FnArg::Receiver(receiver) => {
                    self_receiver = Some(receiver);
                    None
                }
                syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                    syn::Pat::Ident(pat_ident) => {
                        let ident = &pat_ident.ident;
                        Some(quote!(#ident,))
                    }
                    _ => panic!("unsupported pattern argument"),
                },
            })
            .collect::<Vec<_>>();

        quote!(
            #vis #sig {
                self.wrapped.#ident(#(#arguments)*)
            }
        )
    });

    Ok(quote!(
        trait #dyn_ident: 'static + Send {
            #(#trait_methods)*;
        }

        impl #impl_generics #dyn_ident for #self_ty #where_clause {
            #(#methods)*
        }

        #[pymethods]
        impl #py_ident {
            #(#py_methods)*
        }
    ))
}

#[proc_macro_attribute]
pub fn pymethods(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = syn::parse_macro_input!(input as syn::ItemImpl);

    generate_pymethods(item)
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}
