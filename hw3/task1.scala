import org.apache.spark.{SparkConf, SparkContext}
import java.io.PrintWriter
import scala.util.Random
import scala.util.Try

object task1 {
  def mergeDict(x: Map[String, Int], y: Map[String, Int]): Map[String, Int] =
    y.foldLeft(x) { case (acc, (k, v)) => acc + (k -> (acc.getOrElse(k, 0) + v)) }

  def hashF(userIndex: Int, a: Int, b: Int, p: Int, m: Int): Int =
    (((a * userIndex + b) % p) % m)

  def band(id: String, hashValues: List[Int], b: Int, r: Int): List[((Int, List[Int]), List[String])] =
    hashValues
      .grouped(r)
      .zipWithIndex
      .map { case (bandPart, i) => ((i, bandPart), List(id)) }
      .toList

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: scala MainApp <train_file_name> <output_filepath>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Task1")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val input = args(0)
    val output = args(1)
    val csvRDD = sc.textFile(input)
    val startTime = System.currentTimeMillis()

    Try {
      val header = csvRDD.first()
      val rdata = csvRDD.filter(_ != header)
      val data = rdata.map(line => (line.split(",")(1), Map(line.split(",")(0) -> 1)))
      val busDict = data.reduceByKey(mergeDict)
      val uniUser = rdata.map(line => line.split(",")(0)).distinct().zipWithUniqueId().collectAsMap()

      val hashNumber = 50
      val m = busDict.count().toInt
      val p = 100003
      val hashAB = Array.fill(hashNumber)((Random.nextInt(m - 1) + 1, Random.nextInt(m)))

      val userBroadcast = sc.broadcast(uniUser)
      var hashResult = sc.parallelize(Array.empty[((String, Int), Int)])
      hashAB.zipWithIndex.foreach { case ((a, b), index) =>
        val minHash = busDict
          .flatMap { case (businessId, users) =>
            users.keys.map { userId =>
              val userIndex = userBroadcast.value.getOrElse(userId, -1L).toInt
              ((businessId, index), hashF(userIndex, a, b, p, m))
            }
          }
          .reduceByKey(math.min)

        hashResult = if (hashResult.isEmpty()) minHash else hashResult.union(minHash)
      }

      val groupedHash = hashResult
        .map { case ((businessId, index), hashValue) => (businessId, (index, hashValue)) }
        .groupByKey()
        .mapValues(_.toList.sortBy(_._1))
      val finalResult = groupedHash.map { case (businessId, hashes) => (businessId, hashes.map(_._2)) }
      val r = 2
      val b = 25
      val bRdd = finalResult.flatMap { case (businessId, hashValues) => band(businessId, hashValues, b, r) }
      val addR = bRdd.reduceByKey(_ ++ _).filter { case (_, businessList) => businessList.size > 1 }

      val pairs = addR.flatMap { case (_, businessList) =>
        businessList.combinations(2).map(_.sorted).map(pair => (pair(0).toString, pair(1).toString))
      }.distinct().collect().toSet

      val busAgain = busDict.mapValues(_.keys.toSet).collectAsMap()

      val res = pairs.flatMap { case (businessId1, businessId2) =>
        val userSet1 = busAgain.getOrElse(businessId1, Set.empty[String])
        val userSet2 = busAgain.getOrElse(businessId2, Set.empty[String])
        val intersection = userSet1.intersect(userSet2)
        val union = userSet1.union(userSet2)
        val similarity = if (union.nonEmpty) intersection.size.toDouble / union.size else 0.0

        if (similarity >= 0.5) Some(((businessId1, businessId2), similarity)) else None
      }

      val sortedRes = res.toList.sortBy { case ((businessId1, businessId2), _) =>
        (businessId1, businessId2)
      }

      val writer = new PrintWriter(output)
      try {
        writer.println("business_id_1,business_id_2,similarity")
        sortedRes.foreach { case ((businessId1, businessId2), similarity) =>
          writer.println(s"$businessId1,$businessId2,$similarity")
        }
      } finally {
        writer.close()
      }

      val execution = (System.currentTimeMillis() - startTime) / 1000.0
      println(f"Duration: $execution")
    }

    sc.stop()
  }
}
