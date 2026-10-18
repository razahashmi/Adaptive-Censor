// Original MIT setup shell. Uses Windows' installed .NET Framework; bundles no runtime DLLs.
using System;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Security.Cryptography;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using System.Web.Script.Serialization;
using System.Collections.Generic;

class AdaptiveSetup : Form {
    static string Tools, Archive, Expected, Version;
    TextBox folder = new TextBox();
    CheckBox models = new CheckBox();
    Label status = new Label();
    Button setup = new Button(), open = new Button();
    string app;
    static string Q(string value) {
        var result=new StringBuilder("\""); int slashes=0;
        foreach(char c in value) {
            if(c=='\\') {slashes++;continue;}
            if(c=='\"') {result.Append('\\',slashes*2+1);result.Append(c);}
            else {result.Append('\\',slashes);result.Append(c);}
            slashes=0;
        }
        result.Append('\\',slashes*2);result.Append('"');return result.ToString();
    }
    static string Resource(string name) {
        using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name))
        using (var reader = new StreamReader(stream)) return reader.ReadToEnd();
    }
    static int Run(string file, string args, out string output) {
        var start = new ProcessStartInfo(file,args) {UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true};
        start.EnvironmentVariables.Remove("PYTHONHOME");start.EnvironmentVariables.Remove("PYTHONPATH");
        start.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"]="1";
        using (var process = Process.Start(start)) {
            var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
            process.WaitForExit(); Task.WaitAll(stdout,stderr); output=stdout.Result+stderr.Result; return process.ExitCode;
        }
    }
    static string Python() {
        foreach (var command in new string[]{"py","python"}) {
            try {
                string output;
                string args=(command=="py" ? "-3.13 " : "")+"-c "+Q("import sys,struct; print(str(sys.version_info[:2])+'|'+str(struct.calcsize('P')*8)+'|'+sys.executable)");
                if(Run(command,args,out output)==0) {
                    var pieces=output.Trim().Split('|');
                    if(pieces.Length==3 && pieces[0]=="(3, 13)" && pieces[1]=="64") return pieces[2];
                }
            } catch {}
        }
        throw new Exception("Install 64-bit Python 3.13, then reopen this setup helper. The Python download button opens the official website.");
    }
    static void Prepare() {
        var json=new JavaScriptSerializer().Deserialize<Dictionary<string,object>>(Resource("setup-build-info.json"));
        Expected=(string)json["sha256"];Version=(string)json["version"];
        Archive=Path.Combine(AppDomain.CurrentDomain.BaseDirectory,(string)json["archive"]);
        using(var hash=SHA256.Create()) using(var file=File.OpenRead(Archive)) {
            string actual=BitConverter.ToString(hash.ComputeHash(file)).Replace("-","").ToLowerInvariant();
            if(actual!=Expected)throw new Exception("The adjacent source ZIP is missing or changed. Extract the complete reviewed Windows setup ZIP again.");
        }
        Tools=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"AdaptiveCensor","setup-tools",Expected.Substring(0,16));
        Directory.CreateDirectory(Tools);
        foreach(var name in new string[]{"windows_bootstrap.py","build_source_release.py"}) {
            var path=Path.Combine(Tools,name);var bytes=Encoding.UTF8.GetBytes(Resource(name));
            if(File.Exists(path) && !Convert.ToBase64String(File.ReadAllBytes(path)).Equals(Convert.ToBase64String(bytes)))
                throw new Exception("A setup helper file was modified. Choose a new setup version rather than overwriting it.");
            File.WriteAllBytes(path,bytes);
        }
    }
    AdaptiveSetup() {
        Text="Adaptive Censor · Windows CPU setup";ClientSize=new Size(650,420);FormBorderStyle=FormBorderStyle.FixedDialog;MaximizeBox=false;
        Font=new Font("Segoe UI",10);BackColor=Color.FromArgb(245,248,249);
        var title=new Label{Text="Your media. Your rules.",Font=new Font("Segoe UI",23,FontStyle.Bold),Bounds=new Rectangle(26,22,590,50)};
        var intro=new Label{Text="Set up the local CPU preview with Python 3.13.\nDependencies and optional models download separately.",Bounds=new Rectangle(28,88,590,48)};
        folder.Text=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"AdaptiveCensor");folder.SetBounds(28,156,490,30);
        var choose=new Button{Text="Choose…",Bounds=new Rectangle(528,153,94,32)};
        choose.Click+=(s,e)=>{using(var dialog=new FolderBrowserDialog()){dialog.SelectedPath=folder.Text;if(dialog.ShowDialog()==DialogResult.OK)folder.Text=dialog.SelectedPath;}};
        models.Text="Also download base models for analyzing my own files";models.SetBounds(28,208,594,28);
        status.Text="Start with the controls demo; GPU/scene-verifier setup is separate.";status.SetBounds(28,253,594,55);
        setup.Text="Set up";setup.SetBounds(28,325,115,36);
        open.Text="Open player";open.SetBounds(155,325,125,36);open.Enabled=false;
        var python=new Button{Text="Get Python 3.13",Bounds=new Rectangle(450,325,172,36)};
        python.Click+=(s,e)=>Process.Start(new ProcessStartInfo("https://www.python.org/downloads/windows/"){UseShellExecute=true});
        var help=new LinkLabel{Text="Installation guide and support",Bounds=new Rectangle(28,383,340,26)};
        help.Click+=(s,e)=>Process.Start(new ProcessStartInfo("https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/INSTALLATION.md"){UseShellExecute=true});
        setup.Click+=async(s,e)=>{
            setup.Enabled=false;status.Text="Verifying source and installing CPU dependencies. This may take several minutes…";
            string destination=folder.Text;bool withModels=models.Checked;
            try {
                await Task.Run(()=>{
                    Prepare();var pythonPath=Python();string output;
                    string args=Q(Path.Combine(Tools,"windows_bootstrap.py"))+" --install-demo --archive "+Q(Archive)+" --destination "+Q(destination)+" --expected-sha256 "+Expected+" --python "+Q(pythonPath)+(withModels ? " --download-models" : "");
                    if(Run(pythonPath,args,out output)!=0)throw new Exception("Setup could not complete. Check setup.log in the installation folder.\n"+output.Substring(Math.Max(0,output.Length-800)));
                    app=Path.Combine(destination,Version);
                });
                status.Text=withModels ? "Ready. Open the player to review your media." : "Ready. Open the player and try the controls demo.";open.Enabled=true;
            } catch(Exception error){status.Text=error.Message;}
            finally{setup.Enabled=true;}
        };
        open.Click+=async(s,e)=>{
            try {string output="";int code=await Task.Run(()=>Run(Path.Combine(app,".venv","Scripts","python.exe"),Q(Path.Combine(app,"launcher.py"))+" start",out output));if(code!=0)throw new Exception(output);}
            catch(Exception error){MessageBox.Show(this,error.Message,"Could not open player");}
        };
        Controls.AddRange(new Control[]{title,intro,folder,choose,models,status,setup,open,python,help});
    }
    [STAThread] static int Main(string[] args) {
        try {
            if(args.Length>0 && args[0]=="--self-test") {
                Prepare();string output;string rest="";for(int i=1;i<args.Length;i++)rest+=" "+Q(args[i]);
                return Run(Python(),Q(Path.Combine(Tools,"windows_bootstrap.py"))+" --self-test --archive "+Q(Archive)+" --expected-sha256 "+Expected+rest,out output);
            }
            Application.EnableVisualStyles();Application.SetCompatibleTextRenderingDefault(false);Application.Run(new AdaptiveSetup());return 0;
        }catch(Exception error){if(args.Length==0)MessageBox.Show(error.Message,"Adaptive Censor setup");return 1;}
    }
}
