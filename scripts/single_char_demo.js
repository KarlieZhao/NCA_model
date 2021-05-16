<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <title>New CharActers</title>

  <link rel="stylesheet" href="style.css" />
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.3.0/dist/tf.min.js"></script>

</head>

<body>

  <canvas id='canvas' style="display: block; margin-left: 28%; margin-top: 5%; border: 1px solid black; image-rendering: pixelated;"></canvas>

  <select name="selector" id="selector" style="margin-left: 47%;" onchange="setup()">
    <option value="yue_sim">风</option>
    <option value="feng_trad">雅</option>
    <option value="feng_sim">颂</option>

  </select>
  <p style="font-size: 30px; text-align:center; font-style: bold;">
    <span style="font-size: 50px;">N</span>ew <span style="font-size: 50px;">C</span>har<span style="font-size: 50px;">A</span>cters
  </p>
  <p style="text-align:center;">Please wait for the CA to load and generate...<br>
    Shift + LMB to plant a seed</P>
  <script>
    "use strict";
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    const parseConsts = model_graph => {
      const dtypes = {
        'DT_INT32': ['int32', 'intVal', Int32Array],
        'DT_FLOAT': ['float32', 'floatVal', Float32Array]
      };

      const consts = {};
      model_graph.modelTopology.node.filter(n => n.op == 'Const').forEach((node => {
        const v = node.attr.value.tensor;
        const [dtype, field, arrayType] = dtypes[v.dtype];
        if (!v.tensorShape.dim) {
          consts[node.name] = [tf.scalar(v[field][0], dtype)];
        } else {
          // if there is a 0-length dimension, the exported graph json lacks "size"
          const shape = v.tensorShape.dim.map(d => (!d.size) ? 0 : parseInt(d.size));
          let arr;
          if (v.tensorContent) {
            const data = atob(v.tensorContent);
            const buf = new Uint8Array(data.length);
            for (var i = 0; i < data.length; ++i) {
              buf[i] = data.charCodeAt(i);
            }
            arr = new arrayType(buf.buffer);
          } else {
            const size = shape.reduce((a, b) => a * b);
            arr = new arrayType(size);
            if (size) {
              arr.fill(v[field][0]);
            }
          }
          consts[node.name] = [tf.tensor(arr, shape, dtype)];
        }
      }));
      return consts;
    }

    const setup = async () => {
      var files = [];
      files[0] = "feng_sim_12000";
      files[1] = "hua_sim_12000";
      files[2] = "xue_sim_12000";
      files[3] = "yue_sim_12000";

      var consts = [];
      var models = [];

      for (let i = 0; i < files.length; i++) {
        var r = await fetch("outputData/simplified/" + files[i] + ".json");
        consts[i] = parseConsts(await r.json());
        models[i] = await tf.loadGraphModel("outputData/simplified/" + files[i] + ".json");
      }
      Object.assign(models[0].weights, consts[0]);

      // var fileName = document.getElementById("selector").value;
      // const r = await fetch("outputData/" + fileName + ".json");
      // const consts = parseConsts(await r.json());
      //
      // const model = await tf.loadGraphModel("outputData/" + fileName + ".json");
      // Object.assign(model.weights, consts);

      let seed = new Array(16).fill(0).map((x, i) => i < 3 ? 0 : 1);
      seed = tf.tensor(seed, [1, 1, 1, 16]);

      const D = 96 * 2;

      const initState = tf.tidy(() => {
        const D2 = D / 2;
        const a = seed.pad([
          [0, 0],
          [D2 - 1 - 50, 50 + D2], //height
          [D2 - 1, D2], //width
          [0, 0]
        ]); //fill in the array (?
        return a;
      });

      const state = tf.variable(initState);
      const [_, h, w, ch] = state.shape; // => shape of the initialstate tensor: (4,2)

      const damage = (x, y, r) => {
        tf.tidy(() => {
          const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
          const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
          const mask = rx.add(ry).greater(1.0).expandDims(2);
          state.assign(state.mul(mask));
        });
      }

      const plantSeed = (x, y) => {
        const x2 = w - x - seed.shape[2]; //seed => [1,1,1,16]
        const y2 = h - y - seed.shape[1];
        if (x < 0 || x2 < 0 || y2 < 0 || y2 < 0)
          return;
        tf.tidy(() => {
          const a = seed.pad([
            [0, 0],
            [y, y2],
            [x, x2], //(x+x2) has to be (w-seed.shape[2]) which equals to w-1; but if add a to x and subtract it back on x2, the seed will move a cells to the left
            [0, 0]
          ]);
          state.assign(state.add(a));
        });
      }

      const scale = 4;

      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w*scale}px`;
      canvas.style.height = `${h*scale}px`;
      canvas.style.display = 'block';

      canvas.style.marginLeft = "28%";
      canvas.style.marginRight = "5%";

      canvas.onmousedown = e => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((event.clientX - rect.left) / scale);
        const y = Math.floor((event.clientY - rect.top) / scale);
        console.log(x + "," + y);
        if (e.buttons == 1) {
          if (e.shiftKey) {
            plantSeed(x, y);
          } else {
            damage(x, y, 5);
          }
        }
      }
      canvas.onmousemove = e => {
        const rect = canvas.getBoundingClientRect();

        const x = Math.floor((event.clientX - rect.left) / scale);
        const y = Math.floor((event.clientY - rect.top) / scale);
        if (e.buttons == 1 && !e.shiftKey) {
          damage(x, y, 5);
        }
      }
      var stepcount = 0;
      var plantY = h / 2;

      var modelIndex = 0;

      function step() {
        stepcount++;
        if (stepcount % 300 == 0 && stepcount > 100 && stepcount < 1201) {
          console.log(stepcount);
          modelIndex++;
          Object.assign(models[modelIndex].weights, consts[modelIndex]);

        }
        tf.tidy(() => { //tf.tidy() dispose the non-returned results/tensors
          state.assign(models[modelIndex].execute({
            x: state,
            fire_rate: tf.tensor(0.5), //borning rate?
            angle: tf.tensor(0.0), //rotation angle
            step_size: tf.tensor(1.0)
          }, ['Identity']));
        });
      }

      function render() {
        //update every frame
        step();
        // if (stepcount % 300 == 0 && stepcount <= 1200 && stepcount > 100) {
        //   plantY += 15;
        //   console.log(plantY);
        //   plantSeed(w / 2 - 1, plantY);
        // }

        //write in colour data
        const imageData = tf.tidy(() => {
          const rgba = state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
          const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
          const img = tf.tensor(1.0).sub(a).add(rgba).mul(255);
          const rgbaBytes = new Uint8ClampedArray(img.dataSync()); //length: 4*w*h => rgba: why all are 255?


          // process the colour data here to fit our needs
          // for (var i = 0; i < rgbaBytes.length; i++) {
          //   if (rgbaBytes[i] != 255) {
          //     console.log(i + "," + rgbaBytes[i]);
          //   }
          // }
          return new ImageData(rgbaBytes, w, h);
        });

        ctx.putImageData(imageData, 0, 0);
        //-------------

        requestAnimationFrame(render);
      }
      render();

    }
    setup();
  </script>

  <!-- Page footer -->
  <!-- <footer>

    <p style=" position: fixed;left: 0;
   bottom: 0; width: 98%; text-align: right;">
    </p>
  </footer> -->
</body>


</html>
