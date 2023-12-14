use std::{collections::HashMap, fs::File, io::Read};
use std::i32;
use polars::prelude::*;
use rayon::prelude::*;
use extendr_api::{prelude::*, robj};
use extendr_api::wrapper::{Integers, Strings};
use serde_json::*;


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


fn deserialize(filepath: String) -> HashMap<i32, String> {
   
    let mut target = File::open(filepath)
        .expect("Cannot open file. Check file path");
    let mut json_text = String::new();

    target.read_to_string(&mut json_text)
        .expect("Cannot read in json file");


    serde_json::from_str(&json_text)
        .expect("Cannot deserialize json text")
}




fn get_sequence_from_token(vocab_hashmap: HashMap<i32, String>,
                           token: i32) -> String {

    match vocab_hashmap.get(&token) {
       Some(seq) => return seq.to_owned(),
       None => return String::from("")
    }
}


// fn for rust internal work
fn decode(tensor: Vec<i32>,
          vocab_hashmap: HashMap<i32, String>) -> String {
    
    let decoded_seq: String = tensor
        .par_iter()
        .map(|curr_token| get_sequence_from_token(vocab_hashmap.to_owned(), curr_token.to_owned()))
        .collect::<Vec<String>>()
        .join("");

    return decoded_seq;
}


/// @export
#[extendr]
fn decode_tokens(tensor: Integers) -> Strings {

    // reading in the vocab
    let vocab_hashmap: HashMap<i32, String> = deserialize(String::from("./proc/vocab_hashmap.json"));


    let tensor_vec: Vec<i32> = tensor
        .iter()
        .map(|t| i32::from_robj(&Robj::from(t))
             .expect("Cannot convert R int vector to Vec<i32>")
        )
        .collect();

    let decoded_seq: String = decode(tensor_vec, vocab_hashmap);


    return Strings::from(decoded_seq);
}


/// @export
#[extendr]
fn decode_batch(batch: List) -> List {
   
    // reading in the vocab
    let vocab_hashmap: HashMap<i32, String> = deserialize(String::from("./proc/vocab_hashmap.json"));

    let tensor_vec:  Vec<Vec<i32>> = batch
        .iter()
        .map(|t| Robj::from(t.1).as_integer_vector()
             .expect("Cannot convert R int vector to Vec<i32>")
        )
        .collect();

    let decoded_batch: Vec<String> = tensor_vec
        .par_iter()
        .map(|tensor| decode(tensor.to_owned(), vocab_hashmap.to_owned()))
        .collect();

    return List::from_values(decoded_batch);
}


extendr_module! {
   mod tokenizeRs;
   fn decode_tokens;
   fn decode_batch;
}
