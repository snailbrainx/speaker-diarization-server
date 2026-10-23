// Collects the microphone's mono samples into frames of a fixed length and hands each to the page.
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.frameSamples;
    this.frame = new Float32Array(this.size);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const data = input[0];
      for (let i = 0; i < data.length; i++) {
        this.frame[this.n++] = data[i];
        if (this.n === this.size) {
          this.port.postMessage(this.frame, [this.frame.buffer]);
          this.frame = new Float32Array(this.size);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
