
<template>
  <div style="text-align: center">
    <el-container>
      <el-header>
        <form>
          <label for="file-input">选择一个JSON文件:</label>
          <input type="file" id="file-input" name="file-input" accept=".json" />
          <button type="button" @click="loadFile">上传</button>
        </form>
      </el-header>
      <el-container v-show="!isComplete">
        <el-aside>
          <ul v-infinite-scroll="load" class="infinite-list" style="overflow: auto; height: 1000px">
            <li v-for="i in allCount" :key="i" class="infinite-list-item" :style="
              'color:' +
              (i == currentId ? 'red;' : 'blue;') +
              'background-color:' +
              (i == currentId ? '#FFCDCD' : '')
            ">
              {{ i }} : {{ contents[i - 1] }}
            </li>
          </ul>
        </el-aside>
        <el-main>
          <div style="font-size: large">
            当前进度：{{ currentId }}/{{ allCount }}
          </div>

          <div style="font-size: xx-large" v-if="stage == 1">
            {{ bihuas[currentBihua].label }}
          </div>
          <div>
            <el-button type="primary" @click="tiaoguo"> 跳过该字 </el-button>
            <el-button type="primary" @click="restart"> 该字重标 </el-button>
            <el-button type="primary" @click="finishRadical">
              完成打标
            </el-button>
          </div>

          <form>
            <label for="file-input">跳转至第:</label>
            <input id="number-input" type="number" />
            <button type="button" @click="jump">跳转</button>
          </form>

          <svg id="svg"></svg>
          <div>
            左点：<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E5%B7%A6%E7%82%B9.png" width="50"
              height="50">
            斜竖：<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E6%96%9C%E7%AB%96.png" width="50"
              height="50">
            平撇：<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E5%B9%B3%E6%92%87.png" width="50"
              height="20">

          </div>
          <div>
            特殊（开口向上）：<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E5%90%91%E4%B8%8A.png" width="50"
              height="50">
            特殊（开口向下）：<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E5%90%91%E4%B8%8B.png" width="50"
              height="50">
            其他<img src="https://nextlab-static.oss-cn-hangzhou.aliyuncs.com/%E5%85%B6%E4%BB%96.png" width="50"
              height="50">
          </div>


        </el-main>
        <el-aside style="width:320px;">
          <ul style="display: flex;width: 320px;list-style-type: none;flex-flow: wrap;">
            <li v-for="i in bihuas" :key="i" style="width:120px;margin-left:20px;margin-bottom:10px;">
              <el-button style="width:100%;" @click="save(i)"> {{ i.label }} </el-button>
            </li>
          </ul>

        </el-aside>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import { defineComponent } from "vue";

export default defineComponent({
  data() {
    return {
      isComplete: true,
      upm: 0,
      baseline: 0,
      currentId: 1,
      currentIds: 0,
      allCount: 0,
      contents: [],
      contentIds: [],
      contentContours: [],
      save_path: {},
      current_zi_save: {
        ids: "",
        contours: [],
      },
      zi_all_path: [],
      done_all_path: [],
      unsave_zi: {},
      //=====================
      stage: 0,
      currentBihua: 0,
      bihuas: [
        {
          label: "横(一)",
          value: 0,
        },
        {
          label: "竖(丨)",
          value: 1,
        },
        {
          label: "撇(㇁)",
          value: 2,
        },
        {
          label: "捺(㇏)",
          value: 3,
        },
        {
          label: "点(丶)",
          value: 4,
        },
        {
          label: "提(㇀)",
          value: 5,
        },
        {
          label: "横折(𠃍)",
          value: 6,
        },
        {
          label: "横撇(㇇)",
          value: 7,
        },
        {
          label: "横钩(乛)",
          value: 8,
        },
        {
          label: "竖折(㇄)",
          value: 9,
        },
        {
          label: "竖提(𠄌)",
          value: 10,
        },
        {
          label: "竖弯(㇂)",
          value: 11,
        },
        {
          label: "竖钩(亅)",
          value: 12,
        },
        {
          label: "弯钩(㇁)",
          value: 13,
        },
        {
          label: "斜钩(㇂)",
          value: 14,
        },
        {
          label: "撇折(𠃋)",
          value: 15,
        },
        {
          label: "卧钩(㇃)",
          value: 16,
        },
        {
          label: "撇点(𡿨)",
          value: 17,
        },
        {
          label: "横折钩(𠃌)",
          value: 18,
        },
        {
          label: "竖弯钩(乚)",
          value: 19,
        },
        {
          label: "横折弯钩(㇆)",
          value: 20,
        },
        {
          label: "竖折折钩(㇉)",
          value: 21,
        },
        {
          label: "横撇弯钩(㇌)",
          value: 22,
        },
        {
          label: "横折提(㇊)",
          value: 23,
        },
        {
          label: "横折弯(㇍)",
          value: 24,
        },
        {
          label: "横折折折钩(𠄎)",
          value: 25,
        },
        {
          label: "横斜钩(⺄)",
          value: 26,
        },
        {
          label: "横折折撇(㇋)",
          value: 27,
        },
        {
          label: "竖折撇(ㄣ)",
          value: 28,
        },
        {
          label: "竖折折(𠃑)",
          value: 29,
        },
        {
          label: "横折折(㇅)",
          value: 30,
        },
        {
          label: "横折折折(㇎)",
          value: 31,
        },
        {
          label: "左点()",
          value: 32,
        },
        {
          label: "平撇()",
          value: 33,
        },
        {
          label: "斜竖()",
          value: 34,
        },
        {
          label: "特殊（开口向上）",
          value: 35,
        },
        {
          label: "特殊（开口向下）",
          value: 36,
        },
        {
          label: "其他",
          value: 37,
        },
      ],
      a: null,
      current_zi_save_bihua: {
        contours: [],
      },
      save_path_bihua: {},
      unsave_zi_bihua: {},
      svg1_path: [],
      svg2_path: [],
      color: [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#000000",

      ],
      path_arr: []
    };
  },

  mounted() { },
  created() {
    let that = this;
    document.onkeydown = function (e) {
      if (e.keyCode == 13) {

      }
    };
  },
  methods: {
    save(i) {
      console.log(this.save_path_bihua)
      for (var key in this.save_path_bihua) {
        this.path_arr.push({
          "id": this.save_path_bihua[key]["id"],
          "contour": this.save_path_bihua[key]["contour"],
          "type": i.value
        });
        let element = document.getElementById(key)
        element.remove()
      }
      this.save_path_bihua = {}
    },
    restart() {
      this.path_arr = []
      this.done_all_path = []
      this.save_path_bihua = {};
      this.unsave_zi_bihua = {};
      this.draw();
      this.stage = 0;
    },
    tiaoguo() {
      this.path_arr = []
      this.done_all_path = []
      this.save_path_bihua = {};
      this.unsave_zi_bihua = {};
      this.currentId++;
      if (this.currentId == this.allCount + 1) {
        window.alert("该文件所有字体已经打标完成");
        this.isComplete = true;
        this.contents = [];
        this.contentContours = [];
        this.contentIds = [];
        return;
      }
      this.draw();
      this.stage = 0;
    },
    load() { },
    jump() {
      const input = document.getElementById("number-input");
      console.log("input", input);
      this.path_arr = []
      this.done_all_path = []
      this.save_path_bihua = {};
      this.unsave_zi_bihua = {};
      if (input.value >= this.allCount + 1) {
        window.alert("超过最大字数");
        return;
      }
      this.currentId = input.value;
      this.draw();
    },
    finishRadical() {
      console.log(this.done_all_path)
      console.log(this.zi_all_path)
      if (this.zi_all_path.length != this.done_all_path.length) {
        this.$confirm('该字尚有笔画未达标，是否重新打标', '警告', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {
          console.log("restart")
          this.restart()
          this.$message({
            type: 'success',
            message: '重新打标!'
          });
        }).catch(() => {
          console.log("cancel")
          let stroke = {};
          let tmp = {
            upm: this.upm,
            baseline: this.baseline,
            info: this.path_arr
          }
          const blob2 = new Blob([JSON.stringify(tmp)], {
            type: "appication/json",
          });
          // 创建一个<a>标签
          const a2 = document.createElement("a");
          a2.href = URL.createObjectURL(blob2);
          a2.download = `${this.contents[this.currentId - 1]}.json`;

          // 将<a>标签添加到DOM中
          document.body.appendChild(a2);
          a2.click();
          // 清理创建的对象URL
          URL.revokeObjectURL(a2.href);
          this.path_arr = []
          this.done_all_path = []
          this.save_path_bihua = {};
          this.unsave_zi_bihua = {};
          this.currentId++;
          if (this.currentId == this.allCount + 1) {
            window.alert("该文件所有字体已经打标完成");
            this.isComplete = true;
            this.contents = [];
            this.contentContours = [];
            this.contentIds = [];
            return;
          }
          this.draw();
          this.stage = 0;
          return
        })
      } else {
        let stroke = {};
        let tmp = {
          upm: this.upm,
          baseline: this.baseline,
          info: this.path_arr
        }
        const blob2 = new Blob([JSON.stringify(tmp)], {
          type: "appication/json",
        });
        // 创建一个<a>标签
        const a2 = document.createElement("a");
        a2.href = URL.createObjectURL(blob2);
        a2.download = `${this.contents[this.currentId - 1]}.json`;

        // 将<a>标签添加到DOM中
        document.body.appendChild(a2);
        a2.click();
        // 清理创建的对象URL
        URL.revokeObjectURL(a2.href);
        this.path_arr = []
        this.done_all_path = []
        this.save_path_bihua = {};
        this.unsave_zi_bihua = {};
        this.currentId++;
        if (this.currentId == this.allCount + 1) {
          window.alert("该文件所有字体已经打标完成");
          this.isComplete = true;
          this.contents = [];
          this.contentContours = [];
          this.contentIds = [];
          return;
        }
        this.draw();
        this.stage = 0;
      }

    },
    loadFile() {
      let that = this;
      const input = document.getElementById("file-input");
      const file = input.files[0];
      const reader = new FileReader();
      reader.onload = function () {
        const content = reader.result;
        that.initFromContent(content);
        that.isComplete = false;
      };
      reader.readAsText(file);
    },
    initFromContent(content) {
      this.contents = [];
      this.contentContours = [];
      content = JSON.parse(content);
      let upm = content.upm;
      let baseline = content.baseline;
      console.log(upm, baseline);
      this.upm = upm;
      this.baseline = baseline;
      for (var key in content) {
        if (key != "upm" && key != "baseline") {
          console.log(key, content[key]);
          this.contents.push(key);
          this.contentIds.push(content[key].ids);
          this.contentContours.push(content[key].contours);
          this.allCount += 1;
        }
      }
      console.log(this.contentContours);
      this.isComplete = false;
      this.draw();
    },
    draw() {
      for (let i = 0; i < this.zi_all_path.length; i++) {
        this.zi_all_path[i].remove();
      }
      this.zi_all_path = [];
      const svg = document.getElementById("svg");
      let contours = this.contentContours[this.currentId - 1];
      for (let i = 0; i < contours.length; i++) {
        const path = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "path"
        );
        path.setAttribute("id", `path-${i}-${this.currentId}`);
        path.setAttribute("myid", `${i}`)
        path.setAttribute("d", this._mapBezierPointsToSvgPath(contours[i]));
        path.setAttribute("stroke", "black");
        path.setAttribute("stroke-width", "10");
        path.setAttribute("fill", "white");
        svg.append(path);
        this.zi_all_path.push(path);
        path.addEventListener("click", () => {
          if (path.getAttribute("stroke") == "black") {
            path.setAttribute("stroke", "red");
            path.setAttribute("fill", "white");
            this.save_path_bihua[path.getAttribute("id")] = {
              "contour": this.contentContours[this.currentId - 1][
                path.getAttribute("id").split("-")[1]
              ],
              "id": parseInt(path.getAttribute("myid"))
            }
            this.done_all_path.push(path)
          } else {
            path.setAttribute("stroke", "black");
            path.setAttribute("fill", "white");
            delete this.save_path_bihua[path.getAttribute("id")];
            this.done_all_path = this.done_all_path.filter(function (item) {
              return item != path;
            });
          }
        });
      }
    },
    _mapBezierPointsToSvgPath(bezierPoints) {
      let result = `M ${(bezierPoints[0].x / this.upm) * 600},${(1 - (bezierPoints[0].y + this.baseline) / this.upm) * 600
        }`;
      let ends = true;
      let i = 0;
      bezierPoints.forEach((pt) => {
        if (i != 0) {
          result += " ";
          if (pt.on) {
            result +=
              (ends ? "L " : " ") +
              `${(pt.x / this.upm) * 600},${(1 - (pt.y + this.baseline) / this.upm) * 600
              } `;
            ends = true;
          } else {
            result +=
              (ends ? "C " : " ") +
              `${(pt.x / this.upm) * 600},${(1 - (pt.y + this.baseline) / this.upm) * 600
              } `;
            ends = false;
          }
        }
        i = i + 1;
      });
      result +=
        (ends ? "L" : " ") +
        `${(bezierPoints[0].x / this.upm) * 600},${(1 - (bezierPoints[0].y + this.baseline) / this.upm) * 600
        } `;
      return result;
    },
    _mapBezierPointsToSvgPath2(bezierPoints) {
      let result = `M ${(bezierPoints[0].x / this.upm) * 200},${(1 - (bezierPoints[0].y + this.baseline) / this.upm) * 200
        }`;
      let ends = true;
      let i = 0;
      bezierPoints.forEach((pt) => {
        if (i != 0) {
          result += " ";
          if (pt.on) {
            result +=
              (ends ? "L " : " ") +
              `${(pt.x / this.upm) * 200},${(1 - (pt.y + this.baseline) / this.upm) * 200
              } `;
            ends = true;
          } else {
            result +=
              (ends ? "C " : " ") +
              `${(pt.x / this.upm) * 200},${(1 - (pt.y + this.baseline) / this.upm) * 200
              } `;
            ends = false;
          }
        }
        i = i + 1;
      });
      result +=
        (ends ? "L" : " ") +
        `${(bezierPoints[0].x / this.upm) * 200},${(1 - (bezierPoints[0].y + this.baseline) / this.upm) * 200
        } `;
      return result;
    },
  },
});
</script>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}

.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}

#svg {
  width: 600px;
  height: 600px;
  border: 1px #0089a7dd solid;
  border-radius: 8px;
  box-shadow: 2px 2px 16px #0089a766;
  touch-action: none;
  user-select: none;
}

#svg1 {
  width: 200px;
  height: 200px;
  border: 1px #0089a7dd solid;
  border-radius: 8px;
  box-shadow: 2px 2px 16px #0089a766;
  touch-action: none;
  user-select: none;
}

#svg2 {
  width: 200px;
  height: 200px;
  border: 1px #0089a7dd solid;
  border-radius: 8px;
  box-shadow: 2px 2px 16px #0089a766;
  touch-action: none;
  user-select: none;
}


.icon li {
  align-items: center;
  width: 40px;
  margin-bottom: 10px;
}


.infinite-list {
  height: 300px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.infinite-list .infinite-list-item {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 50px;
  background: var(--el-color-primary-light-9);
  margin: 10px;
  color: var(--el-color-primary);
}

.infinite-list .infinite-list-item+.list-item {
  margin-top: 10px;
}
</style>