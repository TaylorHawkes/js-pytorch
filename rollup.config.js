import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import polyfillNode from 'rollup-plugin-polyfill-node'; // Use this if you need to polyfill Node modules

export default {
  input: 'src/index.ts',
  output: {
    file: 'dist/js-pytorch-browser.js',
    format: 'iife', // Change to appropriate format if needed (e.g., umd)
    name: 'JsPytorch',
    globals: {
      fs: 'null' // Ensures `fs` is treated as `null` in the browser
    }
  },
  plugins: [
    resolve({
      browser: true,
      preferBuiltins: false // Important for resolving Node modules to browser versions
    }),
    commonjs(),
    typescript(),
    // Uncomment if you need to polyfill Node built-ins:
    // polyfillNode()
  ]
};
