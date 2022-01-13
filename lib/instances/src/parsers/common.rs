use super::nom_prelude::*;
use std::num::ParseIntError;
use std::str::FromStr;

pub fn separated_list_m_n<I, O, O2, E, F, G>(
  mut min: usize,
  mut max: usize,
  mut sep: G,
  mut f: F,
) -> impl FnMut(I) -> IResult<I, Vec<O>, E>
  where
    I: Clone + PartialEq + nom::InputLength,
    F: nom::Parser<I, O, E>,
    G: nom::Parser<I, O2, E>,
    E: ParseError<I> {
  assert!(min > 0);
  assert!(max >= min);
  min -= 1;
  max -= 1;
  move |input| {
    let (input, mut output) = many_m_n(min, max, terminated(|i| f.parse(i), |i| sep.parse(i))).parse(input)?;
    let (input, last) = f.parse(input)?;
    output.push(last);
    Ok((input, output))
  }
}

pub fn usize_<'a, E>(input: &'a str) -> IResult<&'a str, usize, E>
  where
    E: ParseError<&'a str> + error::FromExternalError<&'a str, ParseIntError>
{
  map_res(digit1, usize::from_str)(input)
}

pub fn isize_<'a, E>(input: &'a str) -> IResult<&'a str, isize, E>
  where
    E: ParseError<&'a str> + error::FromExternalError<&'a str, ParseIntError>
{
  map_res(
    recognize(
      pair(
        opt(char('-')),
        digit1
      )
    ), isize::from_str)(input)
}

pub fn usize_line<'a, E>(input: &'a str) -> IResult<&'a str, usize, E>
  where
    E: ParseError<&'a str> + error::FromExternalError<&'a str, ParseIntError>
{
  terminated(usize_, newline)(input)
}
