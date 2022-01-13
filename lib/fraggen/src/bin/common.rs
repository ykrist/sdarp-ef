use std::fmt::{Debug, Display};
use std::str::FromStr;
use std::path::{PathBuf};
use std::io;
use anyhow::Result;
use structopt::StructOpt;

#[derive(Clone, Debug, StructOpt)]
pub struct OutputOptions {
  #[structopt(long="format", short="f", parse(try_from_str), default_value="json-summ", possible_values=&OUTPUT_FORMAT_STRINGS)]
  pub fmt: OutputFormat,
  #[structopt(long="output", short="o")]
  pub file: Option<PathBuf>,
  #[structopt(long)]
  pub log: Option<PathBuf>,
}

pub fn clap_range_validator<T>(minval: Option<T>, maxval: Option<T>) -> impl Fn(String) -> Result<(), String>
    where
        T: FromStr + PartialOrd + Display + Copy,
        T::Err: Display
{
    return move |val| {
        let x: T = val.parse().map_err(|e: T::Err| e.to_string())?;
        if let Some(y) = minval {
            if x < y { return Err(format!("must be greater than {}", y).to_string()); }
        }
        if let Some(y) = maxval {
            if x > y { return Err(format!("must be less than {}", y).to_string()); }
        }
        return Ok(());
    };
}

pub const OUTPUT_FORMAT_STRINGS: [&str; 2] = ["json", "json-summ"];

#[derive(Debug, Copy, Clone)]
pub enum OutputFormat {
    Json,
    JsonSummary,
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        return match s {
            "json" => Ok(Self::Json),
            "json-summ" => Ok(Self::JsonSummary),
            _ => Err(format!("invalid string: {}", s))
        };
    }
}


impl Default for OutputFormat {
  fn default() -> Self { OutputFormat::JsonSummary }
}


pub trait FragmentGeneration {
    fn write_json(&self, buf : impl io::Write) -> Result<()>;
    fn write_json_summary(&self, buf : impl io::Write) -> Result<()>;

    fn write(&self, buf : impl io::Write, output : OutputFormat) -> Result<()> {
        match output {
            OutputFormat::JsonSummary => self.write_json_summary(buf)?,
            OutputFormat::Json => self.write_json(buf)?,
        };
        Ok(())
    }
}

pub fn output_network(options: &OutputOptions, network: impl FragmentGeneration) -> Result<()> {
  match options.file.as_ref() {
      Some(path) => {
        let writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        network.write(writer, options.fmt)?;
      }
      None => {
        network.write(std::io::stdout(), options.fmt)?;
      }
    }
  Ok(())
}