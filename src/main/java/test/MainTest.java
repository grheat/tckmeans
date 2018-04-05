package test;


import common.TestUtils;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.filter.GenericRecommendedFilter;
import net.librec.filter.RecommendedFilter;
import net.librec.math.algorithm.Randoms;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.similarity.CosineSimilarity;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.util.List;

public class MainTest {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        String s = MainTest.class.getResource("/").toString();
        String resourceDir = s.substring(6, s.length() - 1);
        String dirName = "ml-100k";
        conf.set("dfs.data.dir", resourceDir);
        conf.set("data.input.path", dirName);
        conf.set("data.model.splitter", "testset");
        conf.set("data.testset.path", dirName + "/testData");
        conf.set("data.model.format", "text");
        conf.set("data.column.format", "UIRT");
        conf.set("data.convert.binarize.threshold", "-1.0");
        Randoms.seed(1);

        DataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();

        RecommenderContext context = new RecommenderContext(conf, dataModel);

        conf.set("rec.recommender.similarity.key", "user");
        RecommenderSimilarity similarity = new CosineSimilarity();
        similarity.buildSimilarityMatrix(dataModel);
        context.setSimilarity(similarity);


        TestUtils.testMAEWithKmeans(conf, context, "10");


        /*List recommendedItemList = recommender.getRecommendedList();
        RecommendedFilter filter = new GenericRecommendedFilter();
        recommendedItemList = filter.filter(recommendedItemList);*/

    }
}
