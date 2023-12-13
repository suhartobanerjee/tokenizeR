use std::collections::HashMap;
use std::i32;
use polars::prelude::*;
use rayon::prelude::*;
use extendr_api::prelude::*;
use extendr_api::wrapper::{Integers, Strings};


#[cfg(test)]
mod tests;


fn read_vocab_dt() -> DataFrame {
    //defining the datatypes for the cols
    let myschema = Schema::from_iter(
        vec![
            Field::new("token", DataType::Int32),
            Field::new("sequence", DataType::Utf8)
        ]
    );

    // reading in the vocab dt
    let result = CsvReader::from_path("./proc/bpe_vocab.tsv")
        .expect("File not found!")
        .with_separator("\t".as_bytes()[0])
        .has_header(true)
        .with_schema(Option::Some(Arc::new(myschema)))
        .finish()
        .expect("PolarsError");


    return result;
}


// extract columns and returns as vectors.
fn extract_columns(vocab_dt: DataFrame) -> (Vec<i32>, Vec<String>) {

    let token: Vec<i32> = vocab_dt
        .column("token")
        .unwrap()
        .i32()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();


    let sequence: Vec<String> = vocab_dt
        .column("sequence")
        .expect("Problems in getting the column")
        .utf8()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|seq| seq.to_owned())
        .collect();


    return (token, sequence);
}


fn build_vocab_hashmap(token: Vec<i32>, sequence: Vec<String>) -> HashMap<i32, String> {

    let vocab_hashmap: HashMap<i32, String> = token.iter()
        .zip(sequence.iter())
        .into_iter()
        .map(|x| (x.0.clone(), x.1.clone()))
        .collect();
    

    return vocab_hashmap;
}



fn get_sequence_from_token(vocab_hashmap: HashMap<i32, String>,
                           token: i32) -> String {

    match vocab_hashmap.get(&token) {
       Some(seq) => return seq.to_owned(),
       None => return String::from("")
    }
}


/// @export
#[extendr]
fn decode(tensor: Integers) -> Strings {

    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );

    let vocab_hashmap: HashMap<i32, String> = build_vocab_hashmap(token, sequence);


    let tensor_vec: Vec<i32> = tensor
        .iter()
        .map(|t| i32::from_robj(&Robj::from(t))
             .expect("Cannot convert R int vector to Vec<i32>")
        )
        .collect();

    let decoded_seq: String = tensor_vec
        .par_iter()
        .map(|curr_token| get_sequence_from_token(vocab_hashmap.to_owned(), curr_token.to_owned()))
        .collect::<Vec<String>>()
        .join("");

    return Strings::from(decoded_seq);
}


/// @export
//#[extendr]
//fn decode_batch(batch: Vec<Vec<i32>>,
//                vocab_hashmap: HashMap<i32, String>) -> Vec<String>{
//   
//    let batch_seq: Vec<String> = batch
//        .par_iter()
//        .map(|curr_tensor| decode(curr_tensor.to_owned()))
//        .collect();
//
//    return batch_seq;
//}


extendr_module! {
   mod tokenizeRs;
   fn decode;
//   fn decode_batch;
}
