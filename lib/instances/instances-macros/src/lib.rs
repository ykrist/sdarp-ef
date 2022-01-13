use proc_macro;
use syn::{parse_macro_input, LitStr, Error, Result};
use quote::{quote, ToTokens};
use std::path::{Path, PathBuf};
use proc_macro2::{Span, TokenStream};

fn io_err_to_compile_err(ctx: &str, e: std::io::Error, src: Option<Span>) -> Error {
    let msg = format!("{}: {}", ctx, e);
    Error::new(src.unwrap_or_else(Span::call_site), msg)
}

fn get_path(rel_path: &LitStr) -> Result<PathBuf> {
    let mut p = PathBuf::from(env!("DATA_ROOT"));
    p.push(rel_path.value());
    p.canonicalize()
      .map_err(|e| io_err_to_compile_err("failed to resolve path", e, Some(rel_path.span())))
}

fn get_index_file(dir: &PathBuf) -> Result<PathBuf> {
    let p = dir.join("INDEX.txt");
    p.canonicalize()
      .map_err(|e| io_err_to_compile_err(&format!("failed to locate {:?}", p), e, None))
}

fn parse_index_file(path: &impl AsRef<Path>) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)
      .map_err(|e| io_err_to_compile_err(&format!("failed to read {:?}", path.as_ref()), e, None))?;

    let parsed = contents.split_whitespace().map(|s| s.trim().to_string()).collect();
    Ok(parsed)
}

fn create_new_struct(input: &LitStr) -> Result<TokenStream> {
    let dir = get_path(input)?;
    let index_file = get_index_file(&dir)?;
    let instance_names = parse_index_file(&index_file)?;
    let litstrs : Vec<_> = instance_names.iter().map(|s| LitStr::new(s, Span::call_site())).collect();
    let n = instance_names.len();
    let idxes = 0..n;

    let ts = quote!{
        fn index_to_name(idx: usize) -> &'static str {
            static LOOKUP : [&'static str ; #n] = [#(#litstrs),*];
            LOOKUP[idx]
        }

        fn name_to_index(name: &str) -> usize {
            lazy_static::lazy_static!{
                static ref MAP : std::collections::HashMap<&'static str, usize> = {
                    let mut m = std::collections::HashMap::new();
                    #(m.insert(#litstrs, #idxes));*
                    m
                };
            }
            MAP[name]
        }

    };

    Ok(ts)
}







#[proc_macro]
pub fn std_layout_dataset(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let litstr = parse_macro_input!(input as LitStr);
    println!("{:?}", &litstr);

    match create_new_struct(&litstr) {
        Ok(ts) => ts.into(),
        Err(e) => e.into_compile_error().into()
    }
}
