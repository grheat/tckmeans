package recommender;

import cluster.KMeansClustering;
import net.librec.common.LibrecException;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.util.*;

/**
 * Created by author on 17-11-23.
 */
public class KMeansKNNRecommender extends AbstractRecommender {
    private int knn;
    private DenseVector userMeans;
    private SymmMatrix similarityMatrix;
    private List<Map.Entry<Integer, Double>>[] userSimilarityList;

    /**
     * (non-Javadoc)
     *
     * @see AbstractRecommender#setup()
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        knn = conf.getInt("rec.neighbors.knn.number");
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
    }

    /**
     * (non-Javadoc)
     *
     * @see AbstractRecommender#trainModel()
     */
    @Override
    protected void trainModel() throws LibrecException {
        userMeans = new DenseVector(numUsers);
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            SparseVector userRatingVector = trainMatrix.row(userIdx);
            userMeans.set(userIdx, userRatingVector.getCount() > 0 ? userRatingVector.mean() : globalMean);
        }
    }

    /**
     * (non-Javadoc)
     *
     * @see AbstractRecommender#predict(int, int)
     */
    @Override
    public double predict(int userIdx, int itemIdx) throws LibrecException {
        // create userSimilarityList if not exists
        if (!(null != userSimilarityList && userSimilarityList.length > 0)) {
            createUserSimilarityList();
        }
        // find a number of similar users
        List<Map.Entry<Integer, Double>> nns = new ArrayList<>();
        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx]; //user的相似度排序后的近邻List

        int count = 0;
        Set<Integer> userSet = trainMatrix.getRowsSet(itemIdx);   //对itemId打分的所有用户
        for (Map.Entry<Integer, Double> userRatingEntry : simList) { //遍历近邻用户 如果对Item打过分 得到相似度的值
            int similarUserIdx = userRatingEntry.getKey();  //得到用户userid
            if (!userSet.contains(similarUserIdx)) {
                continue;
            }
            double sim = userRatingEntry.getValue(); //近邻用户对项目评过分 得到相似度
            if (isRanking) {
                nns.add(userRatingEntry);
                count++;
            } else if (sim > 0) {
                nns.add(userRatingEntry);
                count++;
            }
            if (count == knn) {
                break;
            }
        }
        if (nns.size() == 0) {
            return isRanking ? 0 : globalMean;
        }
        if (isRanking) {
            double sum = 0.0d;
            for (Map.Entry<Integer, Double> userRatingEntry : nns) {
                sum += userRatingEntry.getValue();
            }
            return sum;
        }
        // for rating prediction
        double sum = 0, ws = 0;
        for (Map.Entry<Integer, Double> userRatingEntry : nns) {
            int similarUserIdx = userRatingEntry.getKey();
            double sim = userRatingEntry.getValue();
            double rate = trainMatrix.get(similarUserIdx, itemIdx);
            sum += sim * (rate - userMeans.get(similarUserIdx));
            ws += Math.abs(sim);
        }
        return ws > 0 ? userMeans.get(userIdx) + sum / ws : globalMean;
    }


    /**
     * Create userSimilarityList.加上用户分组思想后，需要在该方法上面做修改,取得的是分组后的用户集合<br/>
     * 使用该
     */
    public void createUserSimilarityList() {
        userSimilarityList = new ArrayList[numUsers]; //用户个数
        for (int userIndex = 0; userIndex < numUsers; ++userIndex) {
            // 判断用户属于哪个分组
            // System.out.println("用户Id" + userIndex);
            boolean[] hash = KMeansClustering.getCluster(userIndex);
            SparseVector similarityVector = similarityMatrix.row(userIndex); //userIndex的相似用户相似度
            userSimilarityList[userIndex] = new ArrayList<>(similarityVector.size()); //
            Iterator<VectorEntry> simItr = similarityVector.iterator(); //遍历所有用户相似度
            while (simItr.hasNext()) {
                VectorEntry simVectorEntry = simItr.next();
                if (hash[simVectorEntry.index()] == false)
                    continue;
                // System.out.println(simVectorEntry.index());
                userSimilarityList[userIndex]
                        .add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get())); //在一个聚类里
            }
            Lists.sortList(userSimilarityList[userIndex], true);
        }
    }
}
