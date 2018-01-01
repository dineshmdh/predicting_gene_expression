'''
Created on October 17, 2017

Contains helper functions that will be used to extract the relevant (cell line or tissue) C/T-samples.
'''

import collections as col
import requests
import pandas as pd


def get_df_summary(df):
    '''The df should be a df_metadata, or a df with all necessary fields defined below.'''
    print("Shape:", df.shape)
    print("Number of unique files:", len(set(df['File accession'])))
    print("Number of unique C/T-samples:", len(set(df['Biosample term name'])))
    print("Number of experiment accessions:", len(set(df['Experiment accession'])))
    print("Number of Output type:", len(set(df['Output type'])))
    if (len(set(df['Output type'])) <= 5):
        print("    The output types are:", set(df['Output type']))
    if ('biosample_summary' in df.columns):
        print("Number of unique biosample summaries:", len(set(df['biosample_summary'])))
    else:
        print("There is no biosample_summary field.")


def get_cellline_to_datatype_dict(df_metadata):
    '''Get a dictionary of this format {cell_type : {DNase-seq:5, rnaseq:3}, and so on} to select those cell lines with both data types. With this dict, we can see how many of each datatype is present. This dictionary will be used downstream to subselect the cell lines.
    '''
    cellline_to_datatype = col.OrderedDict()

    for i in range(0, df_metadata.shape[0]):
        afile_info = df_metadata.iloc[i]

        if (afile_info["Biosample organism"] != "Homo sapiens"):
            continue  # << filter for Homo sapiens

        celltype = afile_info["Biosample term name"]
        assay = afile_info["Assay"]

        if not (cellline_to_datatype.keys().__contains__(celltype)):
            cellline_to_datatype[celltype] = {assay: 1}
        else:
            valDict = cellline_to_datatype[celltype]
            if (valDict.keys().__contains__(assay)):
                cellline_to_datatype[celltype][assay] += 1
            else:
                cellline_to_datatype[celltype][assay] = 1

    return cellline_to_datatype


def process_cellline_to_datatype_summary(cellline_to_datatype):
    '''Note that the cellline_to_datatype is a dict of format {cell_type : {DNase-seq:5, rnaseq:3}.
    Print the number of cell lines that have both or either of the data types.
    Also, return the number of "relevant cell lines" - those C/T-samples that have both data types present.
    '''

    '''Get the number of cell types that contain both types of data'''
    cellline_w_2data = 0
    for k, v in cellline_to_datatype.items():
        if len(v.keys()) == 2:  # this can't be more than 2 b/c df_metadata was filtered to have only 2 data types
            cellline_w_2data += 1
            if (cellline_w_2data < 5):
                print(k, v)
    print("..")
    print("Number of unique C/T-samples (or cell lines) w/ both data types (that make relevant_celllines):", cellline_w_2data)
    print("Number of C/T-samples with at least one data type:", len(cellline_to_datatype.items()))

    relevant_celllines = [k for k, v in cellline_to_datatype.items() if (len(v) == 2)]
    return relevant_celllines


def get_celllines_without_dnase_signal(df_metadata, relevant_celllines):
    ''' Some cell lines / tissue samples do not have dnase bigwig signal files. For these, the output types
    available are {'alignments', 'hotspots', 'peaks', 'unfiltered alignments'}. So, for these, will need to
    save the alignment files. (Just a note, this issue is not there for dnase_peaks and rnase_tsvs dfs.)

    Arguments:
    - df_metadata is the barely processed df_metadata file containing both dnase-seq and rnaseq infos.
    - relevant_celllines is a list of all cell lines / tissue samples that have both dnase-seq and rnaseq data
    '''
    dfs = []  # to save dfs that have no bigwig output type
    for act in relevant_celllines:  # act == a cell or tissue
        df = df_metadata[(df_metadata["Biosample term name"] == act) &
                         (df_metadata["File format"] == "bigWig") &
                         (df_metadata["Assay"] == "DNase-seq")]
        if (df.shape[0] == 0):
            df = df_metadata[(df_metadata["Biosample term name"] == act) &
                             (df_metadata["Assay"] == "DNase-seq")]
            dfs.append(df)
    dfs_no_bigwig = pd.concat(dfs)
    return dfs_no_bigwig


def get_biosample_summary(df_meta):
    '''GET 'biosample summary' features from ENCODE server for each experiment accession.
    This can take some time.. (about 5 mins).
    '''
    assert "biosample_summary" not in df_meta.columns
    df_meta["biosample_summary"] = "-"

    for count, anexpt_accession in enumerate(set(df_meta["Experiment accession"])):  # df_gm_dnase_expts
        '''Get the biosample summary'''
        HEADERS = {'accept': 'application/json'}  # Force return from the server in JSON format
        URL = "https://www.encodeproject.org/biosample/" + anexpt_accession + "/?frame=object"
        response = requests.get(URL, headers=HEADERS)  # GET the object
        expt_json_dict = response.json()  # Extract the JSON response as a python dict
        biosample_summary = expt_json_dict["biosample_summary"]

        '''Update the biosample info in the main df'''
        ix = df_meta[df_meta["Experiment accession"] == anexpt_accession].index  # get the indices in the df to update first
        for i in range(0, len(ix)):
            df_meta.set_value(ix[i], "biosample_summary", biosample_summary)

        if (count % 100 == 0):
            print("done with ", count, " experiment accessions.")

    return df_meta


def filter_by_treatment_fraction_phase(df_meta):
    '''Created this function after seeing that Ishikawa cell line has only samples that were treated with some drug.
    Here, it is important that the rnase_tsvs and dnase_signals are treated with the same chemical/drug - if such samples
    are present.

    Note that the df_meta is specific to C/T-sample name.

    Previous version below was very stringent - and not applicable to some cell types:

    reps_before_filter = set(df["Biological replicate(s)"])
    # filtering by biosample treatments, cellular fraction and growth phase
    df = df[[False if (x.__contains__("fraction") or x.__contains__("phase") or x.__contains__("treated with"))
             else True for x in df["biosample_summary"].tolist()]]
    '''
    df_conds = df_meta[["Biosample treatments", "Biosample subcellular fraction term name", "Biosample phase"]]
    df_conds.drop_duplicates(inplace=True)

    df_dnase_signal = df_meta[df_meta["File format"].isin(["bigWig", "bam"])]
    df_rnase_tsvs = df_meta[df_meta["Output type"] == "gene quantifications"]

    for index, row in df_conds.iterrows():  # itertuples():
        trt, frac, phase = row["Biosample treatments"], row["Biosample subcellular fraction term name"], row["Biosample phase"]
        df_dnase_signal_cond = df_dnase_signal[(df_dnase_signal["Biosample treatments"] == trt) &
                                               (df_dnase_signal["Biosample subcellular fraction term name"] == frac) &
                                               (df_dnase_signal["Biosample phase"] == phase)]
        df_rnase_tsvs_cond = df_rnase_tsvs[(df_rnase_tsvs["Biosample treatments"] == trt) &
                                           (df_rnase_tsvs["Biosample subcellular fraction term name"] == frac) &
                                           (df_rnase_tsvs["Biosample phase"] == phase)]

        if (df_dnase_signal_cond.shape[0] > 0) and (df_rnase_tsvs_cond.shape[0] > 0):
            # also filter pks by the same condition tuple if present
            df_dnase_pks = df_meta[df_meta["Output type"] == "peaks"]
            df_dnase_pks_cond = df_dnase_pks[(df_dnase_pks["Biosample treatments"] == trt) &
                                             (df_dnase_pks["Biosample subcellular fraction term name"] == frac) &
                                             (df_dnase_pks["Biosample phase"] == phase)]
            if (df_dnase_pks_cond.shape[0] > 0):
                # print(df_dnase_pks_cond["biosample_summary"])
                # print(df_dnase_signal_cond["biosample_summary"])
                # print(df_rnase_tsvs_cond["biosample_summary"])
                return pd.concat([df_dnase_pks_cond, df_dnase_signal_cond, df_rnase_tsvs_cond])
            else:
                return pd.concat([df_dnase_pks, df_dnase_signal_cond, df_rnase_tsvs_cond])
        else:
            cellline = list(set(df_meta["Biosample term name"]))[0]
            print("WARNING", cellline, "could not be filtered by 'treated with, fraction, phase' conditions.\
              Not one set of condition matched for both dnase signal and rnase tsvs.")
            return df_meta


def filter_by_audit(df, data_type):
    '''data_type can only be one of "dnase_pks", "dnase_signal", "rnase_tsvs"
    Filtering by each replicate since some replicates (within a C/T-sample) can be of
    poor quality and others may not be. If upon filtering, there is no replicate that remains,
    just issue a WARNING.

    Note: Below, df_by_rep is the original/unfiltered version of df_'''
    assert data_type in ["dnase_pks", "dnase_signal", "rnase_tsvs"]
    reps_before_filter = set(df["Biological replicate(s)"])

    dfs_by_reps = []

    for arep in reps_before_filter:  # filtering for each replicate
        df_by_rep = df[df["Biological replicate(s)"] == arep]

        if (data_type == "rnase_tsvs"):
            df_ = df_by_rep[[False if x.__contains__("missing spikeins") else True for x in df_by_rep["Audit WARNING"].tolist()]]
            warning_msg = "WARNING: " + list(set(df["Biosample term name"]))[0] + " has missing spikeins but are retained to save the replicate."
        else:  # for dnase pks or signal
            df_ = df_by_rep[[False if x.__contains__("extremely low spot score") else True for x in df_by_rep["Audit ERROR"].tolist()]]
            warning_msg = "WARNING: " + list(set(df["Biosample term name"]))[0] + " has 'extremely low spot score' but are retained to save the replicate."

        if (df_.shape[0] > 0):
            dfs_by_reps.append(df_)
        else:
            print(warning_msg)
            dfs_by_reps.append(df_by_rep)

    return pd.concat(dfs_by_reps)


def filter_by_library_composition(df, data_type="rnase_tsvs"):
    '''Same as other filtering cases except that this is only relevant for
    data_type "rnase_tsvs"; for others, it is just "DNA". Filter so that the number of replicates are kept
    intact. If upon filtering, the number of replicates decrease, just issue a WARNING.
    '''
    assert data_type == "rnase_tsvs"
    reps_before_filter = set(df["Biological replicate(s)"])

    for madeFrom, depletedIn in [("polyadenylated mRNA", "rRNA"), ("polyadenylated mRNA", "-"), ("RNA", "rRNA"), ("RNA", "-")]:  # in order of preference / precedence
        df_ = df[(df["Library made from"] == madeFrom) & (df["Library depleted in"] == depletedIn)]
        reps_after_filter = set(df_["Biological replicate(s)"])
        if (reps_before_filter == reps_after_filter):
            return df_
    print("WARNING:", list(set(df["Biosample term name"]))[0], "could not be filtered by library composition.")
    return df


def filter_by_lab(df, data_type):
    '''Same as other filtering cases:
    data_type can only be one of "dnase_pks", "dnase_signal", "rnase_tsvs"
    Filter so that the number of replicates are kept intact. If upon filtering,
    the number of replicates decrease, just issue a WARNING.
    By the time this function is called, only 3 labs/pipelines (see below) are present;
    checked with "set(df_metadata["Lab"])".

    Also, similar to filter_by_output_type():
    The filtering is done so that all replicates are of one lab if at all possible (in the order of
    precedence listed below). Else, for each replicate, any of the olabs are selected (also in order of precedence listed below).
    '''
    assert data_type in ["dnase_pks", "dnase_signal", "rnase_tsvs"]
    reps_before_filter = set(df["Biological replicate(s)"])

    for lab in ["ENCODE Processing Pipeline", "John Stamatoyannopoulos, UW", "Gregory Crawford, Duke"]:  # in order of preference / precedence; b/c UW has more DNase data than Duke; spellings checked.
        df_ = df[df["Lab"] == lab]
        reps_after_filter = set(df_["Biological replicate(s)"])
        if (reps_before_filter == reps_after_filter):
            return df_

    print("WARNING:", list(set(df["Biosample term name"]))[0], "could not be filtered by Lab for", data_type, "to have all the replicates of the same output type.")
    dfs_by_reps = []
    for arep in reps_before_filter:
        for lab in ["ENCODE Processing Pipeline", "John Stamatoyannopoulos, UW", "Gregory Crawford, Duke"]:
            df_ = df[(df["Lab"] == lab) & (df["Biological replicate(s)"] == arep)]
            if (df_.shape[0] > 0):
                dfs_by_reps.append(df_)
                break
    return pd.concat(dfs_by_reps)


def filter_by_output_type(df, data_type="dnase_signal"):
    '''Same as other filtering cases except that this is only applicable to
    data_type "dnase_signal". Filter so that the number of replicates are
    kept intact. If upon filtering, the number of replicates decrease, just issue a WARNING.

    The filtering is done so that all replicates are of one output type if at all possible. Else,
    for each replicate, any of the output types are selected (in order of precedence listed below).

    Note that the at this step of filtering, all except for "alignments" correspond to "File format" == "bigWig".
    The "alignments" output type correspond to "bam" files only (this is how the df_metadata was created.) It is
    ensured that any C/T-sample without bigWig file will have a bam alignment file.
    '''
    assert data_type == "dnase_signal"
    reps_before_filter = set(df["Biological replicate(s)"])

    for output_type in ["signal", "raw signal", "signal of unique reads", "base overlap signal", "alignments"]:  # no order of preference / precedence here actually
        df_ = df[df["Output type"] == output_type]
        reps_after_filter = set(df_["Biological replicate(s)"])
        if (reps_before_filter == reps_after_filter):
            return df_

    print("WARNING:", list(set(df["Biosample term name"]))[0], "could not be filtered by output type for dnase bigWigs to have all the replicates of the same output type.")
    dfs_by_reps = []
    for arep in reps_before_filter:
        for output_type in ["signal", "raw signal", "signal of unique reads", "base overlap signal", "alignments"]:
            df_ = df[(df["Output type"] == output_type) & (df["Biological replicate(s)"] == arep)]
            if (df_.shape[0] > 0):
                dfs_by_reps.append(df_)
                break
    return pd.concat(dfs_by_reps)


def pair_tsvs_and_dnase_bigwigs(df_dnase_bigwigs, df_rnase_tsvs):
    '''With all the filtering steps, I make sure that the number of replicates for each output type is kept intact.
    Now, after all the filtering steps, we can pair the rnase_tsvs and dnase bigwigs that can be used downstream in the model.
    '''
    reps_dnase = set(df_dnase_bigwigs["Biological replicate(s)"])
    reps_rnase = set(df_rnase_tsvs["Biological replicate(s)"])

    # cond1: there are as many rows as # unique reps in both data types
    if (df_dnase_bigwigs.shape[0] == df_rnase_tsvs.shape[0] == len(reps_dnase) == len(reps_rnase)):
        return df_dnase_bigwigs, df_rnase_tsvs

    # cond2: rnase has same # rows as # unique reps_rnase, but dnase does not with reps_dnase
    if (len(reps_rnase) == df_rnase_tsvs.shape[0]) and (df_dnase_bigwigs.shape[0] > len(reps_dnase)):
        pass
