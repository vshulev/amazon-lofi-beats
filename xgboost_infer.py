#PSUEDOCODE UNTIL WE GET DATA

def infer(args):
    df = pd.read_csv(args['input_path'], sep='\t')
    model = load_checkpoint()

    y_probs = model.predict_proba(df)
    # topProb = np.argsort(y_probs, axis=1)[:,-3:]
    # topClass = dnamodel.classes_[topProb]

    genuses = {}
    for i in range(len(df)):
        topProbs = np.argsort(y_probs[i], axis=1)[:,-3:]
        topClasses = model.classes_[topProb]

        sampleStr = dnaSeqsEnv['nucraw'][i]
        genuses[sampleStr] = (topClasses, topProbs)

    return genuses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    # parser.add_argument('--checkpt', action='store', type=bool, default=False)

    args = vars(parser.parse_args())