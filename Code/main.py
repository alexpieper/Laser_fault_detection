from data_preprocessing import DataPreprocessing
from data_modelling import NaiveModel, LinearModel, NeuralNetSklearn, ExplainableClassifier
from result_analysis import ModelAnalysis
import dashboard


def main():
    ###############################
    ######## Data Analysis ########
    ###############################

    preprocessing = DataPreprocessing()
    preprocessing.get_stats_of_data()
    preprocessing.visualize_raw_data(n_obs=7)
    
    
    #############################################################
    ######## Data Preprocessing and Train/Test Splitting ########
    #############################################################
    
    training_ratio = 0.5
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                         normalizing=False,
                                                                                         feature_engineering=False)
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                     normalizing=True,
                                                                                                                     feature_engineering=False)
    X_train_feat_eng_raw, X_test_feat_eng_raw, y_train_feat_eng_raw, y_test_feat_eng_raw = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                             normalizing=False,
                                                                                                                             feature_engineering=True,
                                                                                                                             forward_diff=5)
    X_train_feat_eng_normalized, X_test_feat_eng_normalized, y_train_feat_eng_normalized, y_test_feat_eng_normalized = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                                                        normalizing=True,
                                                                                                                                                        feature_engineering=True,
                                                                                                                                                        forward_diff=5)






    ###############################
    ######## Model Creation #######
    ###############################

    # Naive model
    naive_model = NaiveModel(X_train_raw, y_train_raw, 8, 'Naive Model')
    naive_model.train_model()







    # Linear model
    linear_model_raw = LinearModel(X_train_raw, y_train_raw, 'Linear Model (raw)')
    linear_model_raw.train_model()

    linear_model_norm = LinearModel(X_train_normalized, y_train_normalized, 'Linear Model (normalized)')
    linear_model_norm.train_model()

    linear_model_feat = LinearModel(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Linear Model (feature engineered)')
    linear_model_feat.train_model()

    linear_model_norm_feat = LinearModel(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Linear Model (normalized, feature engineered)')
    linear_model_norm_feat.train_model()






    # Explainable model from Microsoft
    explain_ml_raw = ExplainableClassifier(X_train_raw, y_train_raw, 'Explainable Classifier (raw)')
    explain_ml_raw.train_model()
    
    explain_ml_norm = ExplainableClassifier(X_train_normalized, y_train_normalized, 'Explainable Classifier (normalized)')
    explain_ml_norm.train_model()

    explain_ml_feat = ExplainableClassifier(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Explainable Classifier (feature engineered)')
    explain_ml_feat.train_model()

    explain_ml_norm_feat = ExplainableClassifier(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Explainable Classifier (normalized, feature engineered)')
    explain_ml_norm_feat.train_model()








    # MLP model
    ann_model_sklearn_raw = NeuralNetSklearn(X_train_raw, y_train_raw, 'ANN Sklearn (raw)')
    ann_model_sklearn_raw.train_model()

    ann_model_sklearn_norm = NeuralNetSklearn(X_train_normalized, y_train_normalized, 'ANN Sklearn (normalized)')
    ann_model_sklearn_norm.train_model()

    ann_model_sklearn_feat = NeuralNetSklearn(X_train_feat_eng_raw, y_train_feat_eng_raw, 'ANN Sklearn (feature engineered)')
    ann_model_sklearn_feat.train_model()

    ann_model_sklearn_norm_feat = NeuralNetSklearn(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'ANN Sklearn (normalized, feature engineered)')
    ann_model_sklearn_norm_feat.train_model()





    ###############################
    ######## Model Evaluation #####
    ###############################
    ModelAnalysis(X_test_raw, y_test_raw, naive_model).make_whole_report()

    ModelAnalysis(X_test_raw, y_test_raw, linear_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, linear_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, linear_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, linear_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, explain_ml_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, explain_ml_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, explain_ml_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, explain_ml_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, ann_model_sklearn_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, ann_model_sklearn_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, ann_model_sklearn_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, ann_model_sklearn_norm_feat).make_whole_report()


    # Interpretability report
    ModelAnalysis(X_test_raw, y_test_raw, linear_model_raw).make_interpretation_report()
    ModelAnalysis(X_test_raw, y_test_raw, explain_ml_raw).make_interpretation_report()

    

    ###############################################
    ######## Interactive Dashboard Creation #######
    ###############################################

    dashboard.make_evaluation_dashboard(X_test_raw,
                                        y_test_raw,
                                        naive_model,
                                        linear_model_raw,
                                        linear_model_norm,
                                        linear_model_feat,
                                        linear_model_norm_feat,
                                        explain_ml_raw,
                                        explain_ml_norm,
                                        explain_ml_feat,
                                        explain_ml_norm_feat,
                                        ann_model_sklearn_raw,
                                        ann_model_sklearn_norm,
                                        ann_model_sklearn_feat,
                                        ann_model_sklearn_norm_feat,
                                        )

if __name__ == '__main__':
    main()

