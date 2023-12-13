use super::*;

#[test]
fn dt_reading() {
    let vocab_dt = read_vocab_dt();
    println!("{:?}", vocab_dt)
}


#[test]
fn get_columns() {
    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );

    let mut zipped = token.iter()
        .zip(sequence.iter());

    println!("{:?}", zipped.next().unwrap().1)
}


#[test]
fn check_hashmap() {
    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );
    let vocab_hashmap: HashMap<i64, String> = build_vocab_hashmap(token, sequence);

    println!("{:?}", vocab_hashmap.get(&798).unwrap())
}


#[test]
fn get_seq() {
    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );

    let vocab_hashmap: HashMap<i64, String> = build_vocab_hashmap(token, sequence);

    let seq: String = get_sequence_from_token(vocab_hashmap, 2);

    println!("{}", seq);
}


#[test]
fn decode_seq() {
    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );

    let vocab_hashmap: HashMap<i64, String> = build_vocab_hashmap(token, sequence);

    let decoded_seq: String = decode(vec![1, 10, 78, 95, 31_999, 2], vocab_hashmap);

    println!("{}", decoded_seq);
}


#[test]
fn decode_batch_seq() {
    let (token, sequence) = extract_columns(
            read_vocab_dt()
        );

    let vocab_hashmap: HashMap<i64, String> = build_vocab_hashmap(token, sequence);

    let decoded_seq: Vec<String> = decode_batch(vec![vec![1, 8, 9, 17, 2], vec![2, 17, 9, 8, 1]], vocab_hashmap);

    decoded_seq
        .iter()
        .for_each(|seq| println!("{}", seq));
}
