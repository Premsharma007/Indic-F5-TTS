module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt --no-cache"
        ],
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true
          // triton: true
          // sageattention: true
        }
      }
    }
  ]
}
