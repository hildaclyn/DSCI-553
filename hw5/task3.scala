import scala.io.Source
import java.io._
import scala.util.Random
import scala.collection.mutable.ArrayBuffer



object task3 {
  val random = new Random(553) // A single Random instance for use throughout

  def main(args: Array[String]): Unit = {

    val startTime = System.currentTimeMillis()
    val inputs = args(0)
    val streamSize = args(1).toInt
    val asks = args(2).toInt
    val outputs = args(3)

    val bx = new Blackbox()
    var userList = ArrayBuffer.empty[String]
    var total = 0
    val size = 100
    var resultHeader = "seqnum,0_id,20_id,40_id,60_id,80_id\n"

    for (_ <- 1 to asks) {
      val usersStream = bx.ask(inputs, streamSize)
      for  (user <- usersStream) {
        total += 1
        if (userList.length < size) {
          userList += user
        } else {
          if (Random.nextFloat() < 100.0 / total) {
            userList(Random.nextInt(100)) = user
          }
        }
      }
      if (total % 100 == 0) {
        val seqNum = total
        val elements = Seq(seqNum.toString, userList(0), userList(20), userList(40), userList(60), userList(80))
        resultHeader += elements.mkString(",") + "\n"
      }
    }

    // Write the results to a file
    val writer = new PrintWriter(new File(outputs))
    writer.write(resultHeader)
    writer.close()

    val endTime = System.currentTimeMillis()
    println(f"Duration: ${(endTime - startTime) / 1000} seconds")
  }
}
