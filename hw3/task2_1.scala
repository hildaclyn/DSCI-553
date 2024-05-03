import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkConf, SparkContext}

import scala.math._
import java.io.PrintWriter

object task2_1 {
  def mergeD[T](x: Set[T], y: Set[T]): Set[T] = {
    x ++ y
  }
  
  def mergeDict[K, V](x: Map[K, V], y: Map[K, V]): Map[K, V] = {
    x ++ y
  }

  // transform weight based on given logic
  def transformWeight(w: Double, nj: Int): Double = {
    val absW = math.abs(w)
    if (nj <= 2) absW else math.pow(absW, 2)
  }

  // inverse frequency
  def inverseFrequency(n: Int, nj: Int): Double = {
    math.log(n.toDouble / (nj + 1))
  }
  def itembase(use: String, bus: String, uniUse: Set[String], uniBus: Set[String], avgStarU: Map[String, Double], avgStarB: Map[String, Double], useDict: Map[String, Set[String]], busDict: Map[String, Set[String]], orgData: Map[String, Map[String, Double]]): Double = {
    if (!uniUse.contains(use)) return 3
    if (!uniBus.contains(bus)) return avgStarU.getOrElse(use, 3.0)  // Users average or fallback

    val n = uniUse.size  // the number of users for inverse frequency calculation
    val otherBus = useDict.getOrElse(use, Set.empty).filter(line => line != bus)

    val busrating = otherBus.flatMap { bu =>
      val combUse = busDict.getOrElse(bus, Set.empty).intersect(busDict.getOrElse(bu, Set.empty))
      val nj = combUse.size
      val fj = inverseFrequency(n, nj)
      //get the value of inverseFrequency
      val w = (nj match {
        case 0 => 1 - abs(avgStarB.getOrElse(bus, 3.0) - avgStarB.getOrElse(bu, 3.0)) / 2
        case 1 => 1 - abs(orgData.getOrElse(bus, Map.empty).getOrElse(combUse.head, 3.0) - avgStarB.getOrElse(bus, 3.0) - orgData.getOrElse(bu, Map.empty).getOrElse(combUse.head, 3.0) + avgStarB.getOrElse(bu, 3.0)) / 5
        case 2 =>
          val diffs = combUse.map { user =>
            val rI = orgData.getOrElse(bu, Map.empty).getOrElse(user, 3.0) - avgStarU.getOrElse(user, 3.0)
            val rJ = orgData.getOrElse(bus, Map.empty).getOrElse(user, 3.0) - avgStarU.getOrElse(user, 3.0)
            abs(rI - rJ)
          }
          val avgDiff = diffs.sum / diffs.size
          1 - avgDiff / 5  // normalize the result
        case _ => {
          val diffs = combUse.map { user =>
            //calculate the diff for same user diff item
            val ri = orgData(bu)(user) - avgStarB(bu)
            val rj = orgData(bus)(user) - avgStarB(bus)
            val nu = ri * fj * rj * fj
            val denI = pow(ri * fj, 2)
            val denJ = pow(ri * fj, 2)
            (nu, denI, denJ)
          }.foldLeft((0.0, 0.0, 0.0)) { case ((num, di, dj), (nu, denI, denJ)) =>
            (num + nu, di + denI, dj + denJ)
          }
          val den = sqrt(diffs._2) * sqrt(diffs._3)
          if (den != 0) diffs._1 / den else 0
        }
      })
      //transferom the weight
      val wTransformed = transformWeight(w, nj)
      val r = orgData.getOrElse(bu, Map.empty).getOrElse(use, avgStarU.getOrElse(use, 3.0))
      Some((wTransformed, r))
    }

    val (nu, den) = busrating.foldLeft((0.0, 0.0)) { case ((nuc, dec), (w, r)) =>
      (nuc + w * r, dec + abs(w))
    }

    if (den != 0) nu / den else 3.0
  }

  def main(args: Array[String]): Unit = {
    // Check command-line arguments
    if (args.length != 3) {
      println("Usage: scala MainApp <train_file_name> <test_file_name> <output_filepath>")
      System.exit(1)
    }

    val train = args(0)
    val test = args(1)
    val output = args(2)
    //val train = "src/main/resources/yelp_train.csv"
    //val test = "src/main/resources/yelp_val.csv"
    //val output = "src/main/resources/output.csv"

    // Initialize SparkContext
    val conf = new SparkConf().setAppName("Task2_1")
    val sc = new SparkContext(conf)

    // Set log level to ERROR to reduce log output
    sc.setLogLevel("ERROR")

    val startTime = System.currentTimeMillis()

    // load training data
    val trainData = sc.textFile(train)
    val headerTrain = trainData.first()
    val data_nh = trainData.filter(line => line != headerTrain)
    val data_rdd_b = data_nh.map(line => {
      val parts = line.split(',')
      (parts(1), (parts(2).toFloat, 1))
    })
    //combine business and stars calculation
    val comb_star_b = data_rdd_b.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
    val avg_star_b = comb_star_b.mapValues { case (sum, count) => sum / count }.collectAsMap()
    // simlar to the business, we user with stars calculation
    val data_rdd_u = data_nh.map(line => {
      val parts = line.split(',')
      (parts(0), (parts(2).toFloat, 1))
    })

    val comb_star_u = data_rdd_u.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
    val avg_star_u = comb_star_u.mapValues { case (sum, count) => sum / count }.collectAsMap()
    //avg_star_u.foreach(println)
    val avg_star_b_broadcast = sc.broadcast(avg_star_b.mapValues(_.toDouble).toMap)
    val avg_star_u_broadcast = sc.broadcast(avg_star_u.mapValues(_.toDouble).toMap)

    val orgData = data_nh.map(line => {
      val parts = line.split(',')
      (parts(1), Map(parts(0) -> parts(2).toDouble))
    }).reduceByKey(mergeDict).collectAsMap().toMap

    val orgDataB = sc.broadcast(orgData)

    val busDict = data_nh.map(line => {
      val parts = line.split(',')
      (parts(1), Set(parts(0)))
    }).reduceByKey(mergeD).collectAsMap().toMap
    val busDictB = sc.broadcast(busDict)

    val useDict = data_nh.map(line => {
      val parts = line.split(',')
      (parts(0), Set(parts(1)))
    }).reduceByKey(mergeD).collectAsMap().toMap
    val useDictB = sc.broadcast(useDict)
    
    val vals = sc.textFile(test)
    val header = vals.first()
    val clean_val = vals.filter(line => line != header).map(row => {
      val p = row.split(",")
      (p(0), p(1))
    })
    val uniUse = useDictB.value.keys.toSet
    val uniBus = busDictB.value.keys.toSet

    val cal = clean_val.map { case (user, business) =>
      ((user, business), itembase(user, business, uniUse, uniBus, avg_star_u_broadcast.value, avg_star_b_broadcast.value, useDictB.value, busDictB.value, orgDataB.value))
    }
    val results = cal.map { case ((user_id, business_id), stars) =>
      s"$user_id,$business_id,$stars"
    }

    val collectedResults = results.collect()
    //output the file into csv
    val writer = new PrintWriter(output)
    try {
      writer.println("user_id,business_id,prediction")
      collectedResults.foreach { resultString =>
        writer.println(resultString)
      }
    } finally {
      writer.close()
    }

    val endTime = System.currentTimeMillis()
    val duration = (endTime - startTime) / 1000.0
    println(s"Duration: $duration")

    sc.stop()
  }

}
