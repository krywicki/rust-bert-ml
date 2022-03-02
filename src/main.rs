use rust_bert::bart;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use rust_bert::pipelines::sentiment::{Sentiment, SentimentConfig, SentimentModel};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::RustBertError;

type Result<T> = std::result::Result<T, RustBertError>;

fn main() -> Result<()> {
    run_zeroshot()
}

fn run_zeroshot() -> Result<()> {
    let model = zeroshot::model(zeroshot::ModelConfig::BartMNLI)?;

    zeroshot::predict_multilabel(
        model,
        &["product", "customer support", "troubleshooting", "refund"],
        &[
            "I'm calling you today to get some help figuring out how my new router works.",
            "The new washing machine I bought from you guys is no longer working.",
            "Nice weather we're having today.",
            "I bought a motherboard recently and it was DOA. I'm hoping to return this.",
        ],
    );

    Ok(())
}

mod zeroshot {
    use rust_bert::pipelines::zero_shot_classification::{
        ZeroShotClassificationConfig, ZeroShotClassificationModel,
    };

    use super::*;

    pub enum ModelConfig {
        Default,
        BartMNLI,
    }

    pub fn predict_multilabel(
        model: ZeroShotClassificationModel,
        labels: &[&str],
        inputs: &[&str],
    ) {
        let outputs =
            model.predict_multilabel(inputs, labels, Some(Box::new(zeroshot_template)), 128);

        println!("ZeroShot Results");
        println!("============================");
        println!();
        for output in outputs {
            println!("- \"{}\"", inputs[output[0].sentence]);

            for label in output {
                println!("\tlabel: {}", label.text);
                println!("\tscore: {}", label.score);
                println!();
            }
        }
    }

    fn zeroshot_template(label: &str) -> String {
        format!("This conversation is about {}.", label)
    }

    pub fn model(config: ModelConfig) -> Result<ZeroShotClassificationModel> {
        ZeroShotClassificationModel::new(model_config(config))
    }

    pub fn model_config(config: ModelConfig) -> ZeroShotClassificationConfig {
        match config {
            ModelConfig::Default => ZeroShotClassificationConfig::default(),
            ModelConfig::BartMNLI => ZeroShotClassificationConfig::new(
                ModelType::Bart,
                Resource::Remote(RemoteResource::from_pretrained(
                    bart::BartModelResources::BART_MNLI,
                )),
                Resource::Remote(RemoteResource::from_pretrained(
                    bart::BartConfigResources::BART_MNLI,
                )),
                Resource::Remote(RemoteResource::from_pretrained(
                    bart::BartVocabResources::BART_MNLI,
                )),
                Some(Resource::Remote(RemoteResource::from_pretrained(
                    bart::BartMergesResources::BART_MNLI,
                ))),
                false,
                None,
                None,
            ),
        }
    }
}

mod ner {
    use super::*;
}
