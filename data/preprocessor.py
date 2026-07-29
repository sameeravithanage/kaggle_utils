from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class PreprocessorFactory:
    """Factory for building sklearn preprocessing pipelines for tabular data."""

    def __init__(self, conts, nominal_cats, ordinal_mappings):
        """Initialize the preprocessor factory with feature metadata.

        Parameters
        ----------
        conts : list[str]
            Names of continuous columns.
        nominal_cats : list[str]
            Names of nominal categorical columns.
        ordinal_mappings : dict[str, list[str]]
            Mapping of ordinal column names to their ordered category values.
        """

        self.conts = conts
        self.nominal_cats = nominal_cats
        self.ordinal_mappings = ordinal_mappings
        self.preprocessor = None

    def build(self, selected_features, 
              cont_imputer=None, 
              cat_imputer=None):
        """Build and return a preprocessing pipeline for the selected features.

        Parameters
        ----------
        selected_features : list[str] or iterable
            Feature names to include in the preprocessor.
        cont_imputer : sklearn.impute.SimpleImputer, optional
            Imputer to use for continuous features. Defaults to a median imputer.
        cat_imputer : sklearn.impute.SimpleImputer, optional
            Imputer to use for categorical features. Defaults to a constant
            imputer with the fill value ``'Missing'``.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            A configured column transformer that applies the appropriate
            preprocessing steps to the selected continuous, nominal, and
            ordinal features.
        """
        
        # Set default imputers if none are passed
        if cont_imputer is None:
            cont_imputer = SimpleImputer(strategy='median')
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
        
        # 1. Filter the feature lists based on what is in selected_features
        active_conts = [col for col in self.conts if col in selected_features]
        active_nominal = [col for col in self.nominal_cats if col in selected_features and col not in self.ordinal_mappings]
        
        # 2. Filter the ordinal columns AND their mappings together
        active_ordinal = [col for col in self.ordinal_mappings.keys() if col in selected_features]
        active_categories = [self.ordinal_mappings[col] for col in active_ordinal]

        # 3. Build the mini-pipelines dynamically
        transformers = []

        # Only add the continuous pipeline if we actually selected continuous features
        if active_conts:
            num_pipe = Pipeline([('imputer', cont_imputer)])
            transformers.append(('continuous', num_pipe, active_conts))

        # Only add the nominal pipeline if we selected nominal features
        if active_nominal:
            nom_pipe = Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('nominal', nom_pipe, active_nominal))

        # Only add the ordinal pipeline if we selected ordinal features
        if active_ordinal:
            ord_pipe = Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OrdinalEncoder(categories=active_categories, handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('ordinal', ord_pipe, active_ordinal))

        # 4. Assemble and return the final ColumnTransformer
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        
        # Keep pandas dataframe formatting
        preprocessor.set_output(transform="pandas")
        
        self.preprocessor = preprocessor
        return preprocessor

    def show_preprocessor(self):
        """Print the built preprocessor, if one has been created."""
        if self.preprocessor is None:
            print("Preprocessor has not been built yet.")
        else:
            print(self.preprocessor)