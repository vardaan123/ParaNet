import argparse
import sentencepiece as spm

import onmt
from onmt.io.DatasetBase import ONMTDatasetBase
import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts
from onmt.translate import Translator

from argparse import Namespace
import torchtext
from apply_bpe import BPE, read_vocabulary
import subprocess, re
import codecs

from moses_tokenizer import *

class StringONMTDataset(ONMTDatasetBase):
    def __init__(self, src_path, fields):
        self.src_vocabs = []
        self.n_src_feats = 0
        self.n_tgt_feats = 0
        self.data_type = 'text'
        src_base = [{"src": tuple(src_path.split()),
                     "indices": 0}]
        # print(src_base[0]['src'])

        src_examples = (x for x in src_base)
        self.n_src_feats = 0
        examples = src_examples

        # print(self._dynamic_dict)
        # print(type(self._dynamic_dict))

        # examples = self._dynamic_dict(examples)

        ex, examples_iter = self._peek(examples)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (torchtext.data.Example.fromlist(ex_values, fields)
                        for ex_values in example_values)

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(ex_values, out_fields)
            out_examples.append(example)

        super(StringONMTDataset, self).__init__(
            out_examples,
            out_fields,
            None
        )

def translate_opts(parser):
    group = parser.add_argument_group('Model')
    # group.add_argument('-model', required=True,
    #                    help='Path to model .pt file')

    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="Type of the source input. Options: [text|img].")

    # group.add_argument('-src',   required=True,
    #                    help="""Source sequence to decode (one line per
    #                    sequence)""")
    group.add_argument('-src_dir',   default="",
                       help='Source directory for image or audio files')
    group.add_argument('-tgt',
                       help='True target sequence (optional)')
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add_argument('-report_rouge', action='store_true',
                       help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument('-beam_size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=0,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=100,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-coverage_penalty', default='none',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-beta', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    group.add_argument('-block_ngram_repeat', type=int, default=0,
                       help='Block repetition of ngrams during decoding.')
    group.add_argument('-ignore_when_blocking', nargs='+', type=str,
                       default=[],
                       help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=30,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=-1,
                       help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help='Window size for spectrogram in seconds')
    group.add_argument('-window_stride', type=float, default=.01,
                       help='Window stride for spectrogram in seconds')
    group.add_argument('-window', default='hamming',
                       help='Window type for spectrogram generation')


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)

def bpe_decode(input_str_tok):
    # print(subprocess.call("echo {} | sed -r 's/(@@ )|(@@ ?$)//g'".format(input_str)))
    # print('Input str:{}'.format(input_str))
    # return input_str
    out = re.sub('(@@ )|(@@ ?$)','', ' '.join(input_str_tok)).split(' ')
    # print(out)
    # out = input_str_tok
    return out

def paraphrase(input_str, translator_en_de, translator_de_en, sp_en_de, sp_de_en, gpu_id=-1, batch_size=1):
    # --------------------
    # English -> German
    # --------------------
    data_en = StringONMTDataset(input_str, translator_en_de.fields)
    test_data_en = onmt.io.OrderedIterator(
        dataset=data_en, device=gpu_id, batch_size=batch_size, 
        train=False, sort=False, shuffle=False)

    builder_en_de = onmt.translate.TranslationBuilder(
            data_en, translator_en_de.fields,
            kwargs['n_best'], replace_unk=False, has_tgt=False)

    test_data_en.create_batches()
    prepared_string = torchtext.data.Batch(test_data_en.batches[0], data_en, gpu_id, False)

    batch_data, _ = translator_en_de.translate_batch(prepared_string, data_en)
    translations = builder_en_de.from_batch(batch_data)

    trans = translations[0]
    detokenizer = MosesDetokenizer()
        
    # print([sent_piece_model.DecodePieces(pred) for pred in trans.pred_sents[:kwargs['n_best']]])

    n_best_preds = [sp_en_de.DecodePieces(pred) for pred in trans.pred_sents[:kwargs['n_best']]]
    print(n_best_preds)

    trans_de = n_best_preds[0]

    print(trans_de)

    # --------------------
    # German -> English
    # --------------------
    tokenizer = MosesTokenizer()
    trans_de = tokenizer.tokenize(trans_de, return_str=True)
    trans_de = trans_de.lower()
    trans_de = ' '.join(sp_de_en.EncodeAsPieces(trans_de))

    # trans_de = bpe_model.process_line(trans_de)

    data_de = StringONMTDataset(trans_de, translator_de_en.fields)
    test_data_de = onmt.io.OrderedIterator(
        dataset=data_de, device=gpu_id, batch_size=batch_size, 
        train=False, sort=False, shuffle=False)

    builder_de_en = onmt.translate.TranslationBuilder(
            data_de, translator_de_en.fields,
            kwargs['n_best'], replace_unk=False, has_tgt=False)

    test_data_de.create_batches()
    prepared_string = torchtext.data.Batch(test_data_de.batches[0], data_de, gpu_id, False)

    batch_data, _ = translator_de_en.translate_batch(prepared_string, data_de)
    translations = builder_de_en.from_batch(batch_data)

    trans = translations[0]
    detokenizer = MosesDetokenizer()

    n_best_preds = [detokenizer.detokenize(sp_de_en.DecodePieces(x).split(' '), return_str=True, unescape=True) for x in trans.pred_sents[:kwargs['n_best']]]
    # print(n_best_preds)
    trans_en = n_best_preds[0]

    return trans_en

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--spm_model_en_de', type=str, required=True)
    parser.add_argument('--spm_model_de_en', type=str, required=True)
    parser.add_argument('--model_en_de', type=str, required=True)
    parser.add_argument('--model_de_en', type=str, required=True)

    # parser.add_argument('--vocab', type=str, default=None)
    # parser.add_argument('--vocab_thresh', type=int, default=None)

    args = parser.parse_args()

    sp_en_de = spm.SentencePieceProcessor()
    sp_en_de.Load(args.spm_model_en_de)

    # vocabulary = read_vocabulary(codecs.open(args.vocab, 'r', 'utf-8'), args.vocab_thresh)

    sp_de_en = spm.SentencePieceProcessor()
    sp_de_en.Load(args.spm_model_de_en)

    #************************************************************************
    # English -> German
    #************************************************************************
    translate_en_de_parser = argparse.ArgumentParser()
    translate_opts(translate_en_de_parser)
    translator_en_de_args = translate_en_de_parser.parse_known_args([])[0]

    translator_en_de_args.src = 'na'
    translator_en_de_args.model = args.model_en_de
    translator_en_de_args.n_best = 5
    # print(translator_en_de_args)
    
    opt_en_de = translator_en_de_args
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt_en_de = dummy_parser.parse_known_args([])[0]

    fields_en_de, model_en_de, model_opt_en_de = \
        onmt.ModelConstructor.load_test_model(opt_en_de, dummy_opt_en_de.__dict__)

    scorer_en_de = onmt.translate.GNMTGlobalScorer(opt_en_de.alpha,
                                             opt_en_de.beta,
                                             opt_en_de.coverage_penalty,
                                             opt_en_de.length_penalty)

    kwargs = {k: getattr(opt_en_de, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator_en_de = Translator(model_en_de, fields_en_de, global_scorer=scorer_en_de,
                            out_file=None, report_score=True,
                            copy_attn=model_opt_en_de.copy_attn, **kwargs)
    #************************************************************************

    #************************************************************************
    # German -> English
    #************************************************************************
    translate_de_en_parser = argparse.ArgumentParser()
    translate_opts(translate_de_en_parser)
    translator_de_en_args = translate_de_en_parser.parse_known_args([])[0]

    translator_de_en_args.src = 'na'
    translator_de_en_args.model = args.model_de_en
    translator_de_en_args.n_best = 5
    # print(translator_de_en_args)
    
    opt_de_en = translator_de_en_args
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt_de_en = dummy_parser.parse_known_args([])[0]

    fields_de_en, model_de_en, model_opt_de_en = \
        onmt.ModelConstructor.load_test_model(opt_de_en, dummy_opt_de_en.__dict__)

    scorer_de_en = onmt.translate.GNMTGlobalScorer(opt_de_en.alpha,
                                             opt_de_en.beta,
                                             opt_de_en.coverage_penalty,
                                             opt_de_en.length_penalty)

    kwargs = {k: getattr(opt_de_en, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator_de_en = Translator(model_de_en, fields_de_en, global_scorer=scorer_de_en,
                            out_file=None, report_score=True,
                            copy_attn=model_opt_de_en.copy_attn, **kwargs)
    #************************************************************************


    f1 = open(args.input, 'r')
    # f2 = open(args.output, 'w')

    for line in f1:
        input_str = line.rstrip()
        print(input_str)
        input_str_sp = ' '.join(sp_en_de.EncodeAsPieces(input_str))
            
        pphrase = paraphrase(input_str_sp, translator_en_de, translator_de_en, sp_en_de, sp_de_en)

        print(pphrase)
        print('----------')
        # f2.write('{}\n'.format(n_best_preds[0]))

    f1.close()
    # f2.close()
