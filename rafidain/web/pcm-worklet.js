/*
 * رافدَين — AudioWorklet processors for realtime capture & playback.
 * Loaded by app.js via audioWorklet.addModule(). No external deps.
 */

/** Collects 128-sample render quantum frames into fixed-size blocks. */
class RafidainCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.blockSize = opts.blockSize || 2880; // samples @ context rate
    this.buf = new Float32Array(this.blockSize);
    this.n = 0;
    this.muted = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.cmd === "mute") this.muted = !!e.data.value;
    };
  }

  process(inputs) {
    const input = inputs[0];
    const ch = input && input[0];
    if (ch && ch.length) {
      if (this.muted) {
        // keep the block cadence but drop the samples (pause)
        this.n += ch.length;
        while (this.n >= this.blockSize) {
          this.n -= this.blockSize;
        }
      } else {
        for (let i = 0; i < ch.length; i++) {
          this.buf[this.n++] = ch[i];
          if (this.n === this.blockSize) {
            this.port.postMessage(this.buf.slice(0)); // copy out
            this.n = 0;
          }
        }
      }
    }
    return true;
  }
}

/** Drains a jitter queue of Float32 blocks into the output graph. */
class RafidainPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.pos = 0;
    this.underruns = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (!d) return;
      if (d.cmd === "clear") {
        this.queue = [];
        this.pos = 0;
      } else if (d.pcm) {
        this.queue.push(new Float32Array(d.pcm));
        if (this.queue.length > 64) this.queue.splice(0, this.queue.length - 64);
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    let i = 0;
    while (i < out.length && this.queue.length) {
      const cur = this.queue[0];
      const take = Math.min(cur.length - this.pos, out.length - i);
      out.set(cur.subarray(this.pos, this.pos + take), i);
      this.pos += take;
      i += take;
      if (this.pos >= cur.length) {
        this.queue.shift();
        this.pos = 0;
      }
    }
    if (i < out.length) {
      out.fill(0, i);
      this.underruns++;
    }
    this.port.postMessage({ queued: this.queue.length, underruns: this.underruns });
    return true;
  }
}

registerProcessor("rafidain-capture", RafidainCaptureProcessor);
registerProcessor("rafidain-playback", RafidainPlaybackProcessor);
